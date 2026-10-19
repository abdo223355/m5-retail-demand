"""Streamlit web dashboard for M5 Retail Demand Forecasting.

Deployable directly to Streamlit Community Cloud from GitHub.
"""
import sys
from pathlib import Path

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from api.predictor import FeatureContractError, ModelNotLoadedError, Predictor

# Page Configuration
st.set_page_config(
    page_title="M5 Retail Demand Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading LightGBM model artifacts into memory...")
def load_cached_predictor():
    """Initializes and caches the Predictor instance."""
    return Predictor()


# Sample Presets for Fast Testing
PRESETS = {
    "Default Household Item (CA_1)": {
        "item_id": "HOUSEHOLD_1_457", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD",
        "store_id": "CA_1", "state_id": "CA", "wm_yr_wk": 11105, "weekday": "Saturday",
        "wday": 1, "month": 2, "year": 2011, "event_name_1": "No_Event", "event_type_1": "No_Event",
        "event_name_2": "No_Event", "event_type_2": "No_Event", "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
        "sell_price": 9.97, "price_lag_1": 9.97, "price_diff_1": 0.0, "price_pct_change": 0.0,
        "price_max": 10.47, "price_relative_to_max": 0.9522, "lag_1": 0.0, "lag_7": 0.0,
        "lag_14": 0.0, "lag_28": 0.0, "rolling_mean_7": 0.0, "rolling_std_7": 0.0,
        "rolling_mean_14": 0.0, "rolling_std_14": 0.0, "rolling_mean_28": 0.0, "rolling_std_28": 0.0,
        "is_weekend": 0
    },
    "Discounted Item on Weekend": {
        "item_id": "HOUSEHOLD_1_457", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD",
        "store_id": "CA_1", "state_id": "CA", "wm_yr_wk": 11105, "weekday": "Saturday",
        "wday": 1, "month": 2, "year": 2011, "event_name_1": "No_Event", "event_type_1": "No_Event",
        "event_name_2": "No_Event", "event_type_2": "No_Event", "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
        "sell_price": 6.99, "price_lag_1": 9.97, "price_diff_1": -2.98, "price_pct_change": -0.298,
        "price_max": 10.47, "price_relative_to_max": 0.6676, "lag_1": 2.0, "lag_7": 1.0,
        "lag_14": 0.0, "lag_28": 1.0, "rolling_mean_7": 1.43, "rolling_std_7": 0.97,
        "rolling_mean_14": 1.10, "rolling_std_14": 0.85, "rolling_mean_28": 0.75, "rolling_std_28": 0.60,
        "is_weekend": 1
    },
    "High Demand Active Series (TX_2)": {
        "item_id": "HOUSEHOLD_1_457", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD",
        "store_id": "TX_2", "state_id": "TX", "wm_yr_wk": 11210, "weekday": "Sunday",
        "wday": 2, "month": 5, "year": 2012, "event_name_1": "Mother's day", "event_type_1": "Cultural",
        "event_name_2": "No_Event", "event_type_2": "No_Event", "snap_CA": 0, "snap_TX": 1, "snap_WI": 0,
        "sell_price": 8.50, "price_lag_1": 8.50, "price_diff_1": 0.0, "price_pct_change": 0.0,
        "price_max": 10.47, "price_relative_to_max": 0.8118, "lag_1": 6.0, "lag_7": 8.0,
        "lag_14": 5.0, "lag_28": 7.0, "rolling_mean_7": 6.85, "rolling_std_7": 2.10,
        "rolling_mean_14": 6.20, "rolling_std_14": 1.95, "rolling_mean_28": 5.70, "rolling_std_28": 1.80,
        "is_weekend": 1
    }
}


def main():
    st.title("🛒 M5 Retail Demand Prediction Service")
    st.markdown(
        "Interactive machine learning prediction interface serving the trained **LightGBM Regressor** "
        "on Walmart M5 retail sales data."
    )

    predictor = load_cached_predictor()

    # Sidebar: Diagnostics & Presets
    with st.sidebar:
        st.header("⚙️ Model Status")
        if predictor.is_ready():
            st.success("🟢 Model Loaded into Memory")
            st.metric("Total Model Features", len(predictor.feature_names))
            st.caption(f"Categorical Columns: {len(predictor.categorical_cols)}")
        else:
            st.error("🔴 Model Artifacts Not Found")
            st.stop()

        st.markdown("---")
        st.header("⚡ Scenario Presets")
        selected_preset = st.selectbox("Choose a scenario:", list(PRESETS.keys()))
        preset_values = PRESETS[selected_preset]

        st.caption("Selecting a preset automatically populates the form with realistic parameters.")

    # Main Form with Tabbed Categories
    st.subheader("📝 Input Feature Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("##### 🏪 Hierarchy")
        item_id = st.text_input("item_id", value=preset_values["item_id"])
        dept_id = st.selectbox(
            "dept_id",
            ["HOUSEHOLD_1", "HOUSEHOLD_2", "FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2"],
            index=0 if preset_values["dept_id"] == "HOUSEHOLD_1" else 1
        )
        cat_id = st.selectbox(
            "cat_id",
            ["HOUSEHOLD", "FOODS", "HOBBIES"],
            index=0 if preset_values["cat_id"] == "HOUSEHOLD" else 1
        )
        store_id = st.selectbox(
            "store_id",
            ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"],
            index=0 if preset_values["store_id"] == "CA_1" else 5
        )
        state_id = st.selectbox(
            "state_id",
            ["CA", "TX", "WI"],
            index=0 if preset_values["state_id"] == "CA" else 1
        )

    with col2:
        st.markdown("##### 📅 Calendar & Events")
        year = st.number_input("year", min_value=2010, max_value=2030, value=preset_values["year"])
        month = st.number_input("month", min_value=1, max_value=12, value=preset_values["month"])
        wday = st.number_input("wday (1=Sat .. 7=Fri)", min_value=1, max_value=7, value=preset_values["wday"])
        weekday = st.selectbox(
            "weekday",
            ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            index=0 if preset_values["weekday"] == "Saturday" else 1
        )
        wm_yr_wk = st.number_input("wm_yr_wk (Week Index)", value=preset_values["wm_yr_wk"])
        is_weekend = st.selectbox("is_weekend", [0, 1], index=preset_values["is_weekend"])
        event_name_1 = st.text_input("event_name_1", value=preset_values["event_name_1"])
        event_type_1 = st.text_input("event_type_1", value=preset_values["event_type_1"])
        event_name_2 = st.text_input("event_name_2", value=preset_values["event_name_2"])
        event_type_2 = st.text_input("event_type_2", value=preset_values["event_type_2"])
        snap_CA = st.selectbox("snap_CA", [0, 1], index=preset_values["snap_CA"])
        snap_TX = st.selectbox("snap_TX", [0, 1], index=preset_values["snap_TX"])
        snap_WI = st.selectbox("snap_WI", [0, 1], index=preset_values["snap_WI"])

    with col3:
        st.markdown("##### 💰 Pricing")
        sell_price = st.number_input("sell_price ($)", min_value=0.0, value=float(preset_values["sell_price"]), step=0.1)
        price_lag_1 = st.number_input("price_lag_1 ($)", min_value=0.0, value=float(preset_values["price_lag_1"]), step=0.1)
        price_diff_1 = st.number_input("price_diff_1 ($)", value=float(preset_values["price_diff_1"]), step=0.1)
        price_pct_change = st.number_input("price_pct_change", value=float(preset_values["price_pct_change"]), step=0.01)
        price_max = st.number_input("price_max ($)", min_value=0.0, value=float(preset_values["price_max"]), step=0.1)
        price_relative_to_max = st.number_input(
            "price_relative_to_max", min_value=0.0, max_value=1.5,
            value=float(preset_values["price_relative_to_max"]), step=0.01
        )

    with col4:
        st.markdown("##### 📈 Lags & Rolling Stats")
        lag_1 = st.number_input("lag_1 (Sales t-1)", min_value=0.0, value=float(preset_values["lag_1"]), step=1.0)
        lag_7 = st.number_input("lag_7 (Sales t-7)", min_value=0.0, value=float(preset_values["lag_7"]), step=1.0)
        lag_14 = st.number_input("lag_14 (Sales t-14)", min_value=0.0, value=float(preset_values["lag_14"]), step=1.0)
        lag_28 = st.number_input("lag_28 (Sales t-28)", min_value=0.0, value=float(preset_values["lag_28"]), step=1.0)
        rolling_mean_7 = st.number_input("rolling_mean_7", min_value=0.0, value=float(preset_values["rolling_mean_7"]), step=0.1)
        rolling_std_7 = st.number_input("rolling_std_7", min_value=0.0, value=float(preset_values["rolling_std_7"]), step=0.1)
        rolling_mean_14 = st.number_input("rolling_mean_14", min_value=0.0, value=float(preset_values["rolling_mean_14"]), step=0.1)
        rolling_std_14 = st.number_input("rolling_std_14", min_value=0.0, value=float(preset_values["rolling_std_14"]), step=0.1)
        rolling_mean_28 = st.number_input("rolling_mean_28", min_value=0.0, value=float(preset_values["rolling_mean_28"]), step=0.1)
        rolling_std_28 = st.number_input("rolling_std_28", min_value=0.0, value=float(preset_values["rolling_std_28"]), step=0.1)

    payload = {
        "item_id": item_id, "dept_id": dept_id, "cat_id": cat_id, "store_id": store_id, "state_id": state_id,
        "wm_yr_wk": int(wm_yr_wk), "weekday": weekday, "wday": int(wday), "month": int(month), "year": int(year),
        "event_name_1": event_name_1, "event_type_1": event_type_1,
        "event_name_2": event_name_2, "event_type_2": event_type_2,
        "snap_CA": int(snap_CA), "snap_TX": int(snap_TX), "snap_WI": int(snap_WI),
        "sell_price": float(sell_price), "price_lag_1": float(price_lag_1), "price_diff_1": float(price_diff_1),
        "price_pct_change": float(price_pct_change), "price_max": float(price_max),
        "price_relative_to_max": float(price_relative_to_max),
        "lag_1": float(lag_1), "lag_7": float(lag_7), "lag_14": float(lag_14), "lag_28": float(lag_28),
        "rolling_mean_7": float(rolling_mean_7), "rolling_std_7": float(rolling_std_7),
        "rolling_mean_14": float(rolling_mean_14), "rolling_std_14": float(rolling_std_14),
        "rolling_mean_28": float(rolling_mean_28), "rolling_std_28": float(rolling_std_28),
        "is_weekend": int(is_weekend)
    }

    st.markdown("---")

    # Prediction Action
    btn_predict = st.button("🚀 Predict Daily Unit Sales", type="primary", use_container_width=True)

    if btn_predict:
        try:
            with st.spinner("Executing LightGBM inference..."):
                predicted_units = predictor.predict(payload)

            st.balloons()
            st.success("✅ Prediction Completed Successfully!")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(
                    label="🎯 Forecasted Daily Demand",
                    value=f"{predicted_units:.2f} units",
                    help="Predicted unit sales, floored at 0 units (non-negative)."
                )
                st.info(f"Selling Price: **${sell_price:.2f}** | Estimated Daily Revenue: **${predicted_units * sell_price:.2f}**")

            with res_col2:
                st.markdown("##### 📊 Demand Trajectory vs Prediction")
                chart_data = pd.DataFrame({
                    "Time Horizon": ["Day t-28", "Day t-14", "Day t-7", "Day t-1", "Predicted Day t"],
                    "Unit Sales": [lag_28, lag_14, lag_7, lag_1, predicted_units]
                }).set_index("Time Horizon")
                st.line_chart(chart_data)

        except (FeatureContractError, ModelNotLoadedError) as e:
            st.error(f"Validation Error: {e}")
        except Exception as e:
            st.error(f"Inference Failure: {e}")

    # Architectural Overview Footer
    with st.expander("ℹ️ About the Architecture & LightGBM Model"):
        st.markdown("""
        * **Model Architecture:** LightGBM Gradient Boosted Decision Trees (`best_lightgbm_model.pkl`).
        * **Feature Contract:** 34 engineered features with native categorical splitting (`item_id`, `dept_id`, `store_id`, etc.).
        * **Postprocessing:** Strict non-negative clipping `np.clip(prediction, 0, None)`.
        * **Repository Deployment:** Ready for local execution, Docker containers, and Streamlit Community Cloud hosting from GitHub.
        """)


if __name__ == "__main__":
    main()
