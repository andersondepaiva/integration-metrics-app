import io
import sqlite3
from pathlib import Path
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

st.title("📈 Dashboard Integração CRM -> Nótus")
st.caption(
    "Faça upload do CSV gerado pela sua query (colunas esperadas: qtd, status, data_integracao, tipo).\n"
    "O app calcula uma visão do dia atual (com base no último dia do dataset), últimos 7 dias e últimos 30 dias,"
    " e destaca as integrações com mais erros (por tipo)."
)

st.markdown(
    "Query base utilizada (se alterar, mantenha a estrutura):"
)
st.code(
    """select
    count(*) as qtd,
    status,
    data_integracao,
    coalesce(operacao, sis.parent_type) as tipo
from saude_integracao si
inner join saude_integracao_saude_integracao_situacao_1_c sisisc
    on sisisc.saude_integracao_saude_integracao_situacao_1saude_integracao_ida = si.id
    and si.deleted = 0
inner join saude_integracao_situacao sis
    on sisisc.saude_intece37ituacao_idb = sis.id
    and sis.deleted = 0
where data_integracao is not null
    and parent_type is not null
    and tipo = 'envio'
    AND si.data_integracao >= (CURDATE() - INTERVAL 30 DAY)
group by
    status,
    data_integracao,
    coalesce(operacao, sis.parent_type),
    sis.parent_type;
""",
    language="sql"
)
st.markdown("Exporte para CSV e faça o upload no App.")

# Paleta de cores
SUCCESS_COLOR = "#7defa1"   # verde claro
ERROR_COLOR = "#FFABAB"     # vermelho claro (erros)
PARTIAL_COLOR = "#FFFDB8"   # amarelo claro (integrado parcial)
NEUTRAL_COLOR = "#f2f2f2"

# CSS para caixas KPI (inserido uma única vez)
st.markdown(
    f"""
    <style>
    .kpi-box {{
        border-radius: 10px;
        padding: 10px 14px 12px 14px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, rgba(255,255,255,0.65), rgba(255,255,255,0.35));
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        font-family: 'Segoe UI', Arial, sans-serif;
        border: 1px solid rgba(0,0,0,0.05);
        backdrop-filter: blur(4px);
    }}
    .kpi-label {{ font-size: 0.70rem; text-transform: uppercase; letter-spacing: .07em; color:#333; margin-bottom:4px; font-weight:600; }}
    .kpi-value {{ font-size: 1.15rem; font-weight:600; line-height:1.15; color:#111; }}
    .kpi-badge-success {{ background:{SUCCESS_COLOR}; }}
    .kpi-badge-error {{ background:{ERROR_COLOR}; }}
    .kpi-badge-success, .kpi-badge-error {{
        display:inline-block; padding:2px 6px; border-radius:4px; font-size:.65rem; margin-left:6px; vertical-align:middle; color:#111;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.05) inset;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Configuração de persistência
# =============================
DB_PATH = Path("integracoes.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS integracoes (
                status TEXT NOT NULL,
                data_integracao TEXT NOT NULL,
                tipo TEXT NOT NULL,
                qtd INTEGER NOT NULL,
                PRIMARY KEY (status, data_integracao, tipo)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_integracoes_data ON integracoes(data_integracao)")
        conn.commit()


def persist_df(df_csv: pd.DataFrame):
    """Substitui completamente o conteúdo da tabela pelos dados do CSV (truncate + insert).
    Caso o CSV tenha múltiplas linhas com a mesma combinação (status, data_integracao, tipo),
    os valores de qtd são agregados (soma) para evitar violação de PK.
    Retorna (linhas_inseridas, linhas_agrupadas_originalmente).
    """
    # Normaliza datas mesmo se vazio, para garantir consistência
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Limpa base antes de inserir novos dados
        cur.execute("DELETE FROM integracoes")
        if df_csv.empty:
            conn.commit()
            return 0, 0
        rows = df_csv[["status", "data_integracao", "tipo", "qtd"]].copy()
        rows["data_integracao"] = pd.to_datetime(rows["data_integracao"]).dt.strftime("%Y-%m-%d")
        original_count = len(rows)
        # Agrega duplicados
        rows = (
            rows.groupby(["status", "data_integracao", "tipo"], as_index=False)["qtd"].sum()
        )
        data = list(rows.itertuples(index=False, name=None))
        cur.executemany(
            """
            INSERT INTO integracoes (status, data_integracao, tipo, qtd)
            VALUES (?, ?, ?, ?)
            """,
            data,
        )
        conn.commit()
        return len(data), original_count


def load_all_from_db() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["status", "data_integracao", "tipo", "qtd"])  # vazio
    with sqlite3.connect(DB_PATH) as conn:
        df_db = pd.read_sql_query(
            "SELECT status, data_integracao, tipo, qtd FROM integracoes",
            conn,
        )
    if df_db.empty:
        return df_db
    df_db["data_integracao"] = pd.to_datetime(df_db["data_integracao"], errors="coerce")
    df_db = df_db.dropna(subset=["data_integracao"]).copy()
    df_db["dia"] = df_db["data_integracao"].dt.date
    return df_db

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
        or "integrado_parcial" in s
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
    if total > 0:
        sucesso_pct = sucesso / total * 100
        erros_pct = erros / total * 100
    else:
        sucesso_pct = erros_pct = 0.0
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"""
        <div class='kpi-box' style='background:{NEUTRAL_COLOR};'>
            <div class='kpi-label'>Total</div>
            <div class='kpi-value'>{total}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class='kpi-box' style='background:{SUCCESS_COLOR};'>
            <div class='kpi-label'>Sucesso (est.)</div>
            <div class='kpi-value'>{sucesso} <span class='kpi-badge-success'>{sucesso_pct:.1f}%</span></div>
        </div>
        """.replace('.', ','),
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class='kpi-box' style='background:{ERROR_COLOR};'>
            <div class='kpi-label'>Erros (est.)</div>
            <div class='kpi-value'>{erros} <span class='kpi-badge-error'>{erros_pct:.1f}%</span></div>
        </div>
        """.replace('.', ','),
        unsafe_allow_html=True,
    )


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
    # Define cores desejadas: sucesso => verde, integrado_parcial => amarelo, erro => vermelho
    color_map = {}
    for s in agg["status"].unique():
        s_norm = str(s).lower()
        if "integrado_parcial" in s_norm:
            color_map[s] = PARTIAL_COLOR
        elif is_error_status(s_norm):
            color_map[s] = ERROR_COLOR
        else:
            color_map[s] = SUCCESS_COLOR
    fig = px.bar(
        agg,
        x="dia",
        y="qtd",
        color="status",
        text="qtd",
        title=f"Integrações por dia e status — {title_suffix}",
        labels={"dia": "Dia", "qtd": "Quantidade", "status": "Status"},
        color_discrete_map=color_map,
    )
    fig.update_traces(
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
    )
    # Evita poluição visual se houver barras demais
    if len(agg) > 120:
        fig.update_traces(textposition="none")
    # Ordena legenda: Sucessos primeiro (não erro), depois erros
    ordered = sorted(agg["status"].unique(), key=lambda s: (is_error_status(str(s)), str(s)))
    fig.update_layout(legend=dict(traceorder="normal"))
    # Reforça ordem das categorias no eixo/legenda
    fig.update_traces()
    return fig


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
        text="qtd",
        barmode="group",
        title=f"Erros por tipo ao longo do tempo — {title_suffix}",
        labels={"dia": "Dia", "qtd": "Quantidade", "tipo": "Tipo"},
    )
    fig.update_traces(
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
    )
    if len(agg) > 120:
        fig.update_traces(textposition="none")
    return fig, agg


# =============================
# Sidebar — Upload e opções
# =============================
uploaded = st.file_uploader("📤 Faça upload do CSV", type=["csv"]) 

init_db()

df: pd.DataFrame
if uploaded is not None:
    try:
        df_upload = load_csv(uploaded)
    except Exception as ex:
        st.error(f"Falha ao carregar CSV: {ex}")
        st.stop()
    rows_inserted, original_rows = persist_df(df_upload)
    df = load_all_from_db()
    if rows_inserted:
        reducao = original_rows - rows_inserted
        agg_info = f" (agregadas {reducao} linhas duplicadas)" if reducao > 0 else ""
        st.success(f"Upload processado: {rows_inserted} registros inseridos{agg_info}. Total atual: {len(df)}.")
    else:
        st.warning("Base limpa. CSV sem registros válidos.")
else:
    df = load_all_from_db()
    if df.empty:
        st.info("Nenhum dado armazenado ainda. Faça upload de um CSV para popular a base.")
        st.stop()
    else:
        st.success(f"Usando dados já armazenados. Total atual: {len(df)} registros.")

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
max_day = df["dia"].max() if not df.empty else None
if pd.isna(max_day):
    st.warning("Dataset sem datas válidas em data_integracao.")
    st.stop()

st.success(f"Data de referência (último dia armazenado no SQLite): **{max_day}**")

label_hoje = str(max_day)  # mostra a própria data (ISO) em vez de 'Hoje'
masks = {
    label_hoje: window_mask(df, max_day, days=1),
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
        if total > 0:
            sucesso_pct = sucesso / total * 100
            erros_pct = erros / total * 100
        else:
            sucesso_pct = erros_pct = 0.0
        st.markdown(
            f"""
            <div class='kpi-box' style='background:{NEUTRAL_COLOR};'>
                <div class='kpi-label'>Total</div>
                <div class='kpi-value'>{total}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='kpi-box' style='background:{SUCCESS_COLOR};'>
                <div class='kpi-label'>Sucesso</div>
                <div class='kpi-value'>{sucesso} <span class='kpi-badge-success'>{sucesso_pct:.1f}%</span></div>
            </div>
            """.replace('.', ','),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='kpi-box' style='background:{ERROR_COLOR};'>
                <div class='kpi-label'>Erros</div>
                <div class='kpi-value'>{erros} <span class='kpi-badge-error'>{erros_pct:.1f}%</span></div>
            </div>
            """.replace('.', ','),
            unsafe_allow_html=True,
        )

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
st.subheader("Gráficos de Erros por Tipo ")
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


# =============================
# requirements.txt (gerado automaticamente)
# =============================
# Salve este conteúdo em um arquivo separado chamado requirements.txt
#
# streamlit
# pandas
# plotly
