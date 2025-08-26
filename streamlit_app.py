import io
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta, date

# =============================
# Streamlit page configuration
# =============================
st.set_page_config(
    page_title="Visões de Integrações (CSV)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Visões de Integrações a partir de CSV")
st.caption(
    "Faça upload do CSV gerado pela sua query (colunas esperadas: qtd, status, data_integracao, tipo).\n"
    "O app calcula uma visão do dia atual (com base no último dia do dataset), últimos 7 dias e últimos 30 dias,"
    " e destaca as integrações com mais erros (por tipo)."
)

# =============================
# Helpers
# =============================
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    # Tenta detectar separador automaticamente ("," ou ";")
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        file.seek(0)
        df = pd.read_csv(file)
    # Normaliza nomes de colunas
    df.columns = [c.strip().lower() for c in df.columns]

    # Valida colunas mínimas
    required = {"qtd", "status", "data_integracao", "tipo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV não contém as colunas obrigatórias: {', '.join(sorted(missing))}")

    # Converte tipos
    df["qtd"] = pd.to_numeric(df["qtd"], errors="coerce").fillna(0).astype(int)
    df["status"] = df["status"].astype(str)
    df["tipo"] = df["tipo"].astype(str)
    # Garantia de consistência: nada a fazer além das colunas obrigatórias

    # data_integracao -> datetime (sem timezone)
    df["data_integracao"] = pd.to_datetime(df["data_integracao"], errors="coerce")
    df = df.dropna(subset=["data_integracao"]).copy()
    # Apenas a data (para os agrupamentos por dia)
    df["dia"] = df["data_integracao"].dt.date

    return df


def is_error_status(s: str) -> bool:
    s = (s or "").strip().lower()
    # Heurística: marca como erro se contém palavras relacionadas ou for NOK/falha
    return (
        "erro" in s
        or "error" in s
        or "fail" in s
        or s in {"nok", "falha", "failed"}
    )


def _sanitize_pdf_text(text: str) -> str:
    """Remove/normaliza caracteres fora do Latin-1 para evitar FPDFException com fontes core.
    Substitui por '?' quando não suportado."""
    try:
        return text.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return text


def _pdf_safe_multicell(pdf, w, h, text: str):
    from fpdf import FPDFException  # type: ignore
    try:
        pdf.multi_cell(w, h, _sanitize_pdf_text(text))
    except Exception:
        # Tenta dividir em linhas menores caso o erro seja de quebra
        for line in _sanitize_pdf_text(text).splitlines() or [text]:
            try:
                pdf.multi_cell(w, h, line)
            except FPDFException:
                pass


def window_mask(df: pd.DataFrame, max_day: date, days: int) -> pd.Series:
    """Retorna um mask para últimos N dias (incluindo max_day). days=1 => somente max_day."""
    start_day = max_day - timedelta(days=days - 1)
    return (df["dia"] >= start_day) & (df["dia"] <= max_day)


def kpi_row(df_window: pd.DataFrame):
    total = int(df_window["qtd"].sum())
    err_mask = df_window["status"].apply(is_error_status)
    erros = int(df_window.loc[err_mask, "qtd"].sum())
    sucesso = int(total - erros)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de integrações", f"{total:,}".replace(",", "."))
    c2.metric("Sucesso (estimado)", f"{sucesso:,}".replace(",", "."))
    c3.metric("Erros (estimado)", f"{erros:,}".replace(",", "."))


def chart_by_status(df_window: pd.DataFrame, title_suffix: str):
    fig = build_chart_by_status(df_window, title_suffix)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def build_chart_by_status(df_window: pd.DataFrame, title_suffix: str):
    """Retorna fig Plotly de integrações por dia e status."""
    agg = (
        df_window.groupby(["dia", "status"], as_index=False)["qtd"].sum()
        .sort_values("dia")
    )
    return px.bar(
        agg,
        x="dia",
        y="qtd",
        color="status",
        title=f"Integrações por dia e status — {title_suffix}",
        labels={"dia": "Dia", "qtd": "Quantidade"},
    )


def chart_errors_by_tipo(df_window: pd.DataFrame, title_suffix: str, top_n: int = 8):
    fig, agg = build_chart_errors_by_tipo(df_window, title_suffix, top_n)
    if fig is None:
        st.info("Não há linhas de erro para este período.")
        return None
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Ver tabela detalhada (erros por dia x tipo)"):
        st.dataframe(agg, use_container_width=True)
        csv = agg.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV (erros por dia x tipo)",
            data=csv,
            file_name=f"erros_por_dia_tipo_{title_suffix.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            key=f"dl_errors_{title_suffix.replace(' ', '_').lower()}"
        )
    return fig


def build_chart_errors_by_tipo(df_window: pd.DataFrame, title_suffix: str, top_n: int = 8):
    """Retorna (fig, agg) ou (None, None) se não houver erros."""
    err_mask = df_window["status"].apply(is_error_status)
    df_err = df_window.loc[err_mask].copy()
    if df_err.empty:
        return None, None
    top = (
        df_err.groupby("tipo")["qtd"].sum().nlargest(top_n).index.tolist()
    )
    df_top = df_err[df_err["tipo"].isin(top)]
    agg = (
        df_top.groupby(["dia", "tipo"], as_index=False)["qtd"].sum()
        .sort_values(["dia", "qtd"], ascending=[True, False])
    )
    fig = px.bar(
        agg,
        x="dia",
        y="qtd",
        color="tipo",
        barmode="group",
        title=f"Erros por tipo (Top {top_n}) ao longo do tempo — {title_suffix}",
        labels={"dia": "Dia", "qtd": "Quantidade", "tipo": "Tipo"},
    )
    return fig, agg


# =============================
# Sidebar — Upload e opções
# =============================
uploaded = st.file_uploader("📤 Faça upload do CSV", type=["csv"]) 

if uploaded is None:
    st.info(
        "Envie um arquivo .csv com as colunas: **qtd**, **status**, **data_integracao**, **tipo**.\n\n"
    )
    st.stop()

try:
    df = load_csv(uploaded)
except Exception as ex:
    st.error(f"Falha ao carregar CSV: {ex}")
    st.stop()

# Permite ao usuário ajustar a detecção de erro (opcional)
with st.sidebar:
    st.subheader("Configuração de status de erro (opcional)")
    unique_status = sorted(df["status"].astype(str).str.strip().str.lower().unique())
    suggested_errors = [s for s in unique_status if is_error_status(s)]
    selected_errors = st.multiselect(
        "Quais status você considera como erro?",
        options=unique_status,
        default=suggested_errors,
    )

# Se o usuário alterou os erros, reflete na função de máscara
if selected_errors:
    def is_error_status(s: str) -> bool:  # type: ignore[no-redef]
        s = (s or "").strip().lower()
        return s in set(selected_errors)

# =============================
# Janelas de tempo
# =============================
max_day = df["dia"].max()
if pd.isna(max_day):
    st.warning("Dataset sem datas válidas em data_integracao.")
    st.stop()

st.success(f"Data de referência (último dia no dataset): **{max_day}**")

masks = {
    "Hoje (baseado no último dia do dataset)": window_mask(df, max_day, days=1),
    "Últimos 7 dias": window_mask(df, max_day, days=7),
    "Últimos 30 dias": window_mask(df, max_day, days=30),
}

###############################
# Layout comparativo (lado a lado)
###############################
st.header("Comparativo entre janelas de tempo")

# Pré-calcula dataframes por janela
window_dfs = {title: df.loc[mask].copy() for title, mask in masks.items()}

# Linha de KPIs (cada janela em uma coluna)
st.subheader("KPIs")
cols = st.columns(len(window_dfs))
for (title, df_win), col in zip(window_dfs.items(), cols):
    with col:
        st.markdown(f"**{title}**")
        if df_win.empty:
            st.info("Sem dados.")
            continue
        total = int(df_win["qtd"].sum())
        erros = int(df_win.loc[df_win["status"].apply(is_error_status), "qtd"].sum())
        sucesso = total - erros
        st.metric("Total", f"{total:,}".replace(",", "."))
        st.metric("Sucesso (est.)", f"{sucesso:,}".replace(",", "."))
        st.metric("Erros (est.)", f"{erros:,}".replace(",", "."))

# Linha de gráficos por status
st.subheader("Gráficos por Status")
cols_status = st.columns(len(window_dfs))
# Para tentar comparação mais justa, mesma escala Y se possível
max_status_y = 0
status_figs = {}
for title, df_win in window_dfs.items():
    if not df_win.empty:
        agg_tmp = df_win.groupby(["dia", "status"], as_index=False)["qtd"].sum()
        if not agg_tmp.empty:
            max_status_y = max(max_status_y, agg_tmp["qtd"].max())
for (title, df_win), col in zip(window_dfs.items(), cols_status):
    with col:
        st.markdown(f"**{title}**")
        if df_win.empty:
            st.info("Sem dados.")
            continue
        fig = build_chart_by_status(df_win, title)
        if max_status_y > 0:
            fig.update_yaxes(range=[0, max_status_y * 1.05])
        st.plotly_chart(fig, use_container_width=True)
        status_figs[title] = fig

# Linha de gráficos de erros por tipo
st.subheader("Gráficos de Erros por Tipo (Top 8)")
cols_errors = st.columns(len(window_dfs))
max_errors_y = 0
error_figs = {}
for title, df_win in window_dfs.items():
    if not df_win.empty:
        err_fig, agg_err = build_chart_errors_by_tipo(df_win, title)
        if err_fig is not None and agg_err is not None and not agg_err.empty:
            max_errors_y = max(max_errors_y, agg_err["qtd"].max())
        error_figs[title] = (err_fig, agg_err)
for (title, (fig, agg_err)), col in zip(error_figs.items(), cols_errors):
    with col:
        st.markdown(f"**{title}**")
        if fig is None:
            st.info("Sem erros.")
            continue
        if max_errors_y > 0:
            fig.update_yaxes(range=[0, max_errors_y * 1.05])
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Tabela"):
            st.dataframe(agg_err, use_container_width=True)
            csv_err = agg_err.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV (erros)",
                data=csv_err,
                file_name=f"erros_{title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"dl_errors_compare_{title.replace(' ', '_').lower()}"
            )

# Dados brutos comparativos (opcional)
st.subheader("Dados Brutos por Janela")
cols_raw = st.columns(len(window_dfs))
for (title, df_win), col in zip(window_dfs.items(), cols_raw):
    with col:
        st.markdown(f"**{title}**")
        if df_win.empty:
            st.info("Sem dados.")
            continue
        st.dataframe(df_win.sort_values(["dia", "status", "tipo"]), use_container_width=True, height=300)
        csv = df_win.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV",
            data=csv,
            file_name=f"dados_{title.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            key=f"dl_periodo_compare_{title.replace(' ', '_').lower()}"
        )

## (Exportações removidas conforme solicitação)

# =============================
# Rodapé
# =============================
st.markdown(
    "Exemplo de query base (ajuste conforme sua origem de dados):\n"
    "select  count(*) as qtd, status, data_integracao, coalesce(operacao, sis.parent_type) as tipo from saude_integracao si \n"
    "inner join saude_integracao_saude_integracao_situacao_1_c sisisc on sisisc.saude_integracao_saude_integracao_situacao_1saude_integracao_ida = si.id and si.deleted = 0 \n"
    "inner join saude_integracao_situacao sis on sisisc.saude_intece37ituacao_idb = sis.id and sis.deleted = 0 \n"
    "where data_integracao is not null and parent_type is not null \n"
    "and tipo = 'envio' \n"
    "AND si.data_integracao >= (CURDATE() - INTERVAL 30 DAY) \n"
    "group by status, data_integracao, coalesce(operacao, sis.parent_type), sis.parent_type;\n\n"
    "Exporte para CSV e faça o upload no App."
)

# =============================
# requirements.txt (gerado automaticamente)
# =============================
# Salve este conteúdo em um arquivo separado chamado requirements.txt
#
# streamlit
# pandas
# plotly
