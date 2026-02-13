import io
import zipfile
from datetime import datetime, timedelta
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


REQUIRED_GENESYS_COLUMNS = [
    "inin-outbound-id",
    "borrower_id",
    "full_name",
    "phone",
    "CallRecordLastResult-phone",
    "CallRecordLastAgentWrapup-phone",
]

ALLOWED_WRAPUPS = {
    "call back",
    "reject the call",
    "i dont need money",
    "i don't need money",
}

HISTORY_COLUMNS = [
    "id",
    "inin_outbound_id",
    "borrower_id",
    "selected_at",
    "allowed_again_at",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip().strip('"').strip("'") for c in df.columns
    ]
    return df


def normalize_digits(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def first_name(value: object) -> str:
    if pd.isna(value):
        return ""
    parts = str(value).strip().split()
    return parts[0] if parts else ""


def to_excel_bytes(df: pd.DataFrame, header: bool = True) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, header=header)
    buffer.seek(0)
    return buffer.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8").encode("utf-8")


def load_history_df(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.read_csv(uploaded_file, dtype=str, encoding="utf-8")
    history = normalize_columns(history)

    missing = [c for c in HISTORY_COLUMNS if c not in history.columns]
    if missing:
        raise ValueError(
            "El archivo historico no tiene el formato esperado. "
            f"Columnas faltantes: {missing}"
        )

    history = history[HISTORY_COLUMNS].copy()

    history["selected_at"] = pd.to_datetime(history["selected_at"], errors="coerce").dt.date
    history["allowed_again_at"] = pd.to_datetime(
        history["allowed_again_at"], errors="coerce"
    ).dt.date

    invalid_selected = int(history["selected_at"].isna().sum())
    invalid_allowed = int(history["allowed_again_at"].isna().sum())
    if invalid_selected > 0 or invalid_allowed > 0:
        raise ValueError(
            "El historico contiene fechas invalidas en 'selected_at' o 'allowed_again_at'."
        )

    history["id_num"] = pd.to_numeric(history["id"], errors="coerce")
    if history["id_num"].isna().any():
        raise ValueError("El historico contiene valores invalidos en la columna 'id'.")

    history["id"] = history["id_num"].astype("int64").astype(str)
    history = history.drop(columns=["id_num"])

    history["borrower_id"] = history["borrower_id"].astype(str).str.strip()
    history["inin_outbound_id"] = history["inin_outbound_id"].astype(str).str.strip()

    return history


def compute_outputs(
    genesys_file,
    history_file,
) -> Tuple[Dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    now = datetime.now(ZoneInfo("America/Mexico_City"))
    today = now.date()
    run_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    genesys_df = pd.read_csv(genesys_file, dtype=str)
    genesys_df = normalize_columns(genesys_df)

    missing_genesys = [c for c in REQUIRED_GENESYS_COLUMNS if c not in genesys_df.columns]
    if missing_genesys:
        raise ValueError(
            "El archivo Genesys no contiene todas las columnas requeridas. "
            f"Faltan: {missing_genesys}"
        )

    history_df = load_history_df(history_file)

    blocked_borrowers = set(
        history_df.loc[history_df["allowed_again_at"] > today, "borrower_id"].astype(str)
    )

    df = genesys_df.copy()
    df["source_filename"] = genesys_file.name
    df["campaign_run_at"] = run_ts

    df["borrower_id_norm"] = df["borrower_id"].astype(str).str.strip()
    df["inin_outbound_id"] = df["inin-outbound-id"].astype(str).str.strip()

    result_raw = df["CallRecordLastResult-phone"].fillna("").astype(str)
    wrap_raw = df["CallRecordLastAgentWrapup-phone"].fillna("").astype(str)

    df["filter_cooldown_pass"] = ~df["borrower_id_norm"].isin(blocked_borrowers)
    df["filter_result_pass"] = result_raw.eq("") | result_raw.str.startswith("ININ-OUTBOUND")

    wrap_norm = wrap_raw.str.strip().str.lower()
    df["filter_wrapup_pass"] = wrap_norm.eq("") | wrap_norm.isin(ALLOWED_WRAPUPS)

    df["phone_raw"] = df["phone"].fillna("").astype(str)
    df["phone_full"] = df["phone_raw"].apply(normalize_digits)
    df["phone_10"] = df["phone_full"].str[-10:]
    df["filter_phone_pass"] = df["phone_10"].str.len().eq(10)

    df["first_name"] = df["full_name"].apply(first_name)

    df["is_selected_final"] = (
        df["filter_cooldown_pass"]
        & df["filter_result_pass"]
        & df["filter_wrapup_pass"]
        & df["filter_phone_pass"]
    )

    selected_at = today.isoformat()
    allowed_again_at = (today + timedelta(days=7)).isoformat()

    df["selected_at"] = ""
    df["allowed_again_at"] = ""
    selected_mask = df["is_selected_final"]
    df.loc[selected_mask, "selected_at"] = selected_at
    df.loc[selected_mask, "allowed_again_at"] = allowed_again_at

    # Priority: cooldown -> result -> wrapup -> invalid_phone -> selected
    df["drop_reason"] = "selected"
    df.loc[~df["filter_cooldown_pass"], "drop_reason"] = "cooldown"
    df.loc[df["filter_cooldown_pass"] & ~df["filter_result_pass"], "drop_reason"] = "result"
    df.loc[
        df["filter_cooldown_pass"] & df["filter_result_pass"] & ~df["filter_wrapup_pass"],
        "drop_reason",
    ] = "wrapup"
    df.loc[
        df["filter_cooldown_pass"]
        & df["filter_result_pass"]
        & df["filter_wrapup_pass"]
        & ~df["filter_phone_pass"],
        "drop_reason",
    ] = "invalid_phone"

    selected_df = df[df["is_selected_final"]].copy()

    campaign_df = pd.DataFrame(
        {
            "country_code": "52",
            "phone": selected_df["phone_10"],
            "first_name": selected_df["first_name"],
        }
    )
    if len(campaign_df) == 0:
        campaign_df = pd.DataFrame(columns=["country_code", "phone", "first_name"])

    # Build official historical rows for selected records only.
    new_history_rows = selected_df[
        ["inin_outbound_id", "borrower_id_norm", "selected_at", "allowed_again_at"]
    ].rename(columns={"borrower_id_norm": "borrower_id"})
    new_history_rows["borrower_id"] = new_history_rows["borrower_id"].astype(str).str.strip()
    new_history_rows["inin_outbound_id"] = (
        new_history_rows["inin_outbound_id"].astype(str).str.strip()
    )
    new_history_rows = new_history_rows.drop_duplicates(
        subset=["inin_outbound_id", "borrower_id", "selected_at"], keep="first"
    )

    existing_keys = set(
        history_df[["inin_outbound_id", "borrower_id", "selected_at"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )

    if len(new_history_rows) > 0:
        new_history_rows["is_existing"] = new_history_rows.apply(
            lambda r: (
                str(r["inin_outbound_id"]),
                str(r["borrower_id"]),
                str(r["selected_at"]),
            )
            in existing_keys,
            axis=1,
        )
        new_history_rows = new_history_rows[~new_history_rows["is_existing"]].drop(
            columns=["is_existing"]
        )

    max_id = int(pd.to_numeric(history_df["id"], errors="coerce").max()) if len(history_df) else 0

    if len(new_history_rows) > 0:
        new_history_rows = new_history_rows.copy()
        new_history_rows.insert(
            0,
            "id",
            [str(v) for v in range(max_id + 1, max_id + len(new_history_rows) + 1)],
        )
        new_history_rows = new_history_rows[HISTORY_COLUMNS]
    else:
        new_history_rows = pd.DataFrame(columns=HISTORY_COLUMNS)

    combined_history = pd.concat(
        [history_df[HISTORY_COLUMNS], new_history_rows[HISTORY_COLUMNS]], ignore_index=True
    )
    combined_history = combined_history.drop_duplicates(
        subset=["inin_outbound_id", "borrower_id", "selected_at"], keep="first"
    )

    # Keep date columns as ISO strings for CSV output consistency.
    combined_history["selected_at"] = combined_history["selected_at"].astype(str)
    combined_history["allowed_again_at"] = combined_history["allowed_again_at"].astype(str)

    audit_df = df[
        [
            "source_filename",
            "campaign_run_at",
            "borrower_id_norm",
            "inin_outbound_id",
            "full_name",
            "phone_raw",
            "phone_full",
            "phone_10",
            "first_name",
            "filter_cooldown_pass",
            "filter_result_pass",
            "filter_wrapup_pass",
            "filter_phone_pass",
            "is_selected_final",
            "drop_reason",
            "selected_at",
            "allowed_again_at",
        ]
    ].rename(columns={"borrower_id_norm": "borrower_id"})

    metrics = {
        "registros_entrada": int(len(df)),
        "removidos_cooldown": int((~df["filter_cooldown_pass"]).sum()),
        "removidos_result": int((df["filter_cooldown_pass"] & ~df["filter_result_pass"]).sum()),
        "removidos_wrapup": int(
            (
                df["filter_cooldown_pass"]
                & df["filter_result_pass"]
                & ~df["filter_wrapup_pass"]
            ).sum()
        ),
        "removidos_phone": int(
            (
                df["filter_cooldown_pass"]
                & df["filter_result_pass"]
                & df["filter_wrapup_pass"]
                & ~df["filter_phone_pass"]
            ).sum()
        ),
        "seleccionados_finales": int(df["is_selected_final"].sum()),
        "nuevos_en_historico": int(len(new_history_rows)),
        "registros_campana": int(len(campaign_df)),
    }

    run_date = now.strftime("%d-%m-%Y")
    run_hhmm = now.strftime("%H%M")

    return metrics, combined_history, audit_df, campaign_df, run_date, run_hhmm


def build_final_zip(
    history_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    campaign_df: pd.DataFrame,
    run_date: str,
    run_hhmm: str,
) -> Tuple[str, bytes]:
    date_token = run_date.lower()
    zip_name = f"telesales_campaign_{run_date}.zip"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        campaign_name = f"1_campaign_wa_{date_token}.xlsx"
        zf.writestr(campaign_name, to_excel_bytes(campaign_df, header=False))

        hist_name = f"sent_history_{date_token}.csv"
        zf.writestr(hist_name, to_csv_bytes(history_df))

        audit_name = f"audit_campaign_{date_token}-{run_hhmm}.xlsx"
        zf.writestr(audit_name, to_excel_bytes(audit_df, header=True))

    buffer.seek(0)
    return zip_name, buffer.getvalue()


def build_empty_history_template() -> bytes:
    template = pd.DataFrame(columns=HISTORY_COLUMNS)
    return to_csv_bytes(template)


def main() -> None:
    st.set_page_config(page_title="Telesales WA Generator", layout="wide")

    st.title("Telesales - Generador de Campana WA + Auditoria")

    st.info(
        "Esta herramienta NO consulta ClickHouse ni APIs. "
        "Todo se procesa con archivos cargados por el usuario."
    )

    st.warning(
        "IMPORTANTE: Esta plantilla solo genera informacion para plantillas de WhatsApp "
        "con una sola variable: nombre."
    )

    st.markdown(
        """
### Como usar esta UI
1. Descarga desde Genesys el CSV de Telesales del dia.
2. Carga tambien el historico mas reciente que tengas, llamado `sent_history_dd-mm-yyyy.csv`.
3. Si no lo encuentras, puedes continuar sin historico y la app creara uno nuevo con la data del dia.
4. Haz clic en **Generar paquete final (.zip)** y descarga el ZIP unico con los 3 entregables.
"""
    )

    st.markdown(
        """
### Que obtendras al final
- `1_campaign_wa_dd-mm-yyyy.xlsx`: archivo para cargar WhatsApp (sin encabezado).
- `sent_history_dd-mm-yyyy.csv`: historico oficial para Risk/Analytics. Debes usar siempre el mas reciente.
- `audit_campaign_dd-mm-yyyy-hhmm.xlsx`: snapshot completo del dia con filtros y motivo de descarte/seleccion.
"""
    )

    st.download_button(
        "Descargar plantilla CSV de historico obligatorio",
        data=build_empty_history_template(),
        file_name="PLANTILLA_CSV_HISTORICO_OBLIGATORIO.csv",
        mime="text/csv",
    )

    st.divider()

    genesys_file = st.file_uploader(
        "Sube el archivo Genesys CSV (obligatorio)",
        type=["csv"],
        accept_multiple_files=False,
    )

    history_file = st.file_uploader(
        "Sube el CSV historico acumulado (opcional)",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.caption(
        "Recomendado: sube el archivo `sent_history_...` mas reciente para respetar cooldown de 7 dias y evitar reenvios a clientes ya contactados por WhatsApp."
    )
    st.warning(
        "Si no subes el historico, la app no podra bloquear clientes contactados en los ultimos 7 dias y podrias volver a enviarles mensaje."
    )

    with st.expander("Reglas aplicadas", expanded=False):
        st.markdown(
            """
- **Cooldown**: se bloquea borrower_id con `allowed_again_at > hoy` usando el historico subido.
- **Result**: `CallRecordLastResult-phone` vacio o que empiece por `ININ-OUTBOUND`.
- **Wrapup** (literal): vacio o uno de:
  - `call back`
  - `reject the call`
  - `i dont need money`
  - `i don't need money`
- **Telefono**: se dejan solo ultimos 10 digitos validos.
- **Campana WA**: salida en un solo archivo, sin encabezados.
"""
        )

    with st.expander("Como leer el archivo de auditoria", expanded=False):
        st.markdown(
            """
- `filter_cooldown_pass`: `True` si NO esta en cooldown.
- `filter_result_pass`: `True` si cumple la regla de resultado de llamada.
- `filter_wrapup_pass`: `True` si cumple la regla de wrapup permitido.
- `filter_phone_pass`: `True` si el telefono queda valido a 10 digitos.
- `is_selected_final`: `True` si paso todos los filtros y si entra a campana.
- `drop_reason`: motivo final.
  - `selected`: paso todos los filtros.
  - `cooldown`: se bloqueo por historial reciente.
  - `result`: no paso el filtro de resultado de llamada.
  - `wrapup`: no paso el filtro de wrapup.
  - `invalid_phone`: telefono invalido tras normalizacion.
"""
        )

    if st.button("Generar paquete final (.zip)", type="primary"):
        if genesys_file is None:
            st.error("Debes subir el archivo Genesys CSV para continuar.")
            return

        try:
            (
                metrics,
                history_updated,
                audit_df,
                campaign_df,
                run_date,
                run_hhmm,
            ) = compute_outputs(
                genesys_file=genesys_file,
                history_file=history_file,
            )

            zip_name, zip_bytes = build_final_zip(
                history_df=history_updated,
                audit_df=audit_df,
                campaign_df=campaign_df,
                run_date=run_date,
                run_hhmm=run_hhmm,
            )

            st.success("Proceso completado. Ya puedes descargar el paquete final.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registros entrada", metrics["registros_entrada"])
            c2.metric("Seleccionados finales", metrics["seleccionados_finales"])
            c3.metric("Nuevos en historico", metrics["nuevos_en_historico"])
            c4.metric("Registros campana", metrics["registros_campana"])

            c5, c6, c7 = st.columns(3)
            c5.metric("Removidos cooldown", metrics["removidos_cooldown"])
            c6.metric("Removidos result", metrics["removidos_result"])
            c7.metric("Removidos wrapup", metrics["removidos_wrapup"])

            st.metric("Removidos telefono invalido", metrics["removidos_phone"])

            st.download_button(
                "Descargar paquete final (.zip)",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
            )

        except Exception as exc:
            st.error(f"No se pudo generar el paquete: {exc}")


if __name__ == "__main__":
    main()
