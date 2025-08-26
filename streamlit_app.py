import io
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
    "Faça upload do CSV gerado pela sua query (colunas esperadas: qtd, status, data_integracao, tipo, parent_type).\n"
    "O app calcula uma visão do dia atual (com base no último dia do dataset), últimos 7 dias e últimos 30 dias,"
    " e destaca as integrações com mais erros (por parent_type)."
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

    # Renomeia se vier como "sis.parent_type"
    if "sis.parent_type" in df.columns and "parent_type" not in df.columns:
        df = df.rename(columns={"sis.parent_type": "parent_type"})

    # Valida colunas mínimas
    required = {"qtd", "status", "data_integracao", "tipo", "parent_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV não contém as colunas obrigatórias: {', '.join(sorted(missing))}")

    # Converte tipos
    df["qtd"] = pd.to_numeric(df["qtd"], errors="coerce").fillna(0).astype(int)
    df["status"] = df["status"].astype(str)
    df["tipo"] = df["tipo"].astype(str)
    df["parent_type"] = df["parent_type"].astype(str)

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
    # Agrega por dia + status somando a qtd, pois o CSV já vem agregado na origem
    agg = (
        df_window.groupby(["dia", "status"], as_index=False)["qtd"].sum()
        .sort_values("dia")
    )
    fig = px.bar(
        agg,
        x="dia",
        y="qtd",
        color="status",
        title=f"Integrações por dia e status — {title_suffix}",
        labels={"dia": "Dia", "qtd": "Quantidade"},
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_errors_by_parent_type(df_window: pd.DataFrame, title_suffix: str, top_n: int = 8):
    # Filtra somente linhas de erro
    err_mask = df_window["status"].apply(is_error_status)
    df_err = df_window.loc[err_mask].copy()
    if df_err.empty:
        st.info("Não há linhas de erro para este período.")
        return

    # Top N parent_types por soma de erros no período
    top = (
        df_err.groupby("parent_type")["qtd"].sum().nlargest(top_n).index.tolist()
    )
    df_top = df_err[df_err["parent_type"].isin(top)]

    agg = (
        df_top.groupby(["dia", "parent_type"], as_index=False)["qtd"].sum()
        .sort_values(["dia", "qtd"], ascending=[True, False])
    )

    fig = px.bar(
        agg,
        x="dia",
        y="qtd",
        color="parent_type",
        barmode="group",
        title=f"Erros por parent_type (Top {top_n}) ao longo do tempo — {title_suffix}",
        labels={"dia": "Dia", "qtd": "Quantidade", "parent_type": "Parent Type"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabela detalhada (erros por dia x parent_type)"):
        st.dataframe(agg, use_container_width=True)
        csv = agg.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV (erros por dia x parent_type)",
            data=csv,
            file_name=f"erros_por_dia_parent_type_{title_suffix.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            key=f"dl_errors_{title_suffix.replace(' ', '_').lower()}"
        )


# =============================
# Sidebar — Upload e opções
# =============================
uploaded = st.file_uploader("📤 Faça upload do CSV", type=["csv"]) 

if uploaded is None:
    st.info(
        "Envie um arquivo .csv com as colunas: **qtd**, **status**, **data_integracao**, **tipo**, **parent_type**.\n\n"
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

# =============================
# Abas por janela
# =============================
aba_hoje, aba_7d, aba_30d = st.tabs(list(masks.keys()))
abas = [aba_hoje, aba_7d, aba_30d]

for (title, mask), tab in zip(masks.items(), abas):
    with tab:
        df_win = df.loc[mask].copy()
        st.subheader(title)
        if df_win.empty:
            st.info("Sem dados para este período.")
            continue

        # KPIs
        kpi_row(df_win)

        # Gráfico por status
        chart_by_status(df_win, title_suffix=title)

        # Erros por parent_type
        chart_errors_by_parent_type(df_win, title_suffix=title, top_n=8)

        # Tabela base do período (opcional)
        with st.expander("Ver dados brutos do período"):
            st.dataframe(df_win.sort_values(["dia", "status", "parent_type"]), use_container_width=True)
            csv = df_win.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV do período",
                data=csv,
                file_name=f"dados_{title.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key=f"dl_periodo_{title.replace(' ', '_').lower()}"
            )

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
