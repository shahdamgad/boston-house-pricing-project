"""
Boston Housing Price Prediction - Streamlit app

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     learning_curve, train_test_split,
                                     validation_curve)
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------------------ setup --
st.set_page_config(page_title="Boston Housing Price Prediction",
                   page_icon="🏡", layout="wide")

# CSS uses translucent colours so it looks right in both light and dark themes
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .main-title {font-size: 38px; font-weight: 700; color: #1b9ac4; margin-bottom: 0;}
    .subtitle {font-size: 18px; opacity: 0.7;}
    .prediction-card {background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.5);
                      padding: 25px; border-radius: 15px; text-align: center;}
    .prediction-card h1 {color: #22c55e; margin: 0;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.12); padding: 15px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

TARGET = "MEDV"
FEATURE_INFO = {
    "RM": "Average number of rooms per home",
    "LSTAT": "% of homeowners in the neighborhood considered 'lower class'",
    "PTRATIO": "Students per teacher in nearby schools",
}
# The three clients from the project's prediction question
CLIENTS = {
    "Client 1": {"RM": 5, "LSTAT": 17, "PTRATIO": 15},
    "Client 2": {"RM": 4, "LSTAT": 32, "PTRATIO": 22},
    "Client 3": {"RM": 8, "LSTAT": 3, "PTRATIO": 12},
}


def money(v: float) -> str:
    return f"${v:,.0f}"


def rgba(rgb: tuple, a: float) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


RED, GREEN = (239, 68, 68), (34, 197, 94)


# ------------------------------------------------------------------- data --
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


# ------------------------------------------------------------------ model --
def fit_best_tree(X, y):
    """Grid search over max_depth 1-10 with ShuffleSplit CV, scored by R²."""
    cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    grid = GridSearchCV(DecisionTreeRegressor(random_state=0),
                        {"max_depth": list(range(1, 11))},
                        scoring="r2", cv=cv)
    grid.fit(X, y)
    return grid


@st.cache_resource
def train_model(data: pd.DataFrame, features: list):
    X, y = data[features], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    grid = fit_best_tree(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    return {
        "model": model, "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "cv_score": grid.best_score_,
    }


@st.cache_data
def complexity_curve(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    depths = np.arange(1, 11)
    train, valid = validation_curve(
        DecisionTreeRegressor(random_state=0), X, y, param_name="max_depth",
        param_range=depths, cv=cv, scoring="r2")
    return depths, train, valid


@st.cache_data
def learning_curves(X, y, depth: int):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    sizes = np.rint(np.linspace(1, X.shape[0] * 0.8 - 1, 9)).astype(int)
    sizes, train, valid = learning_curve(
        DecisionTreeRegressor(max_depth=depth, random_state=0), X, y,
        cv=cv, train_sizes=sizes, scoring="r2")
    return sizes, train, valid


@st.cache_data
def prediction_trials(X, y, row: tuple, features: list):
    """Fit the tuned model on 10 different train/test splits and predict one home."""
    preds = []
    for k in range(10):
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=k)
        best = fit_best_tree(X_tr, y_tr).best_estimator_
        preds.append(float(best.predict(pd.DataFrame([row], columns=features))[0]))
    return preds


def add_band(fig, x, scores, name, color):
    mean, std = scores.mean(axis=1), scores.std(axis=1)
    fig.add_trace(go.Scatter(x=x, y=mean + std, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mean - std, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=rgba(color, 0.15),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines+markers", name=name,
                             line=dict(color=rgba(color, 1))))


# ------------------------------------------------------------ sidebar/data --
st.sidebar.title("🏡 Boston Housing")
st.sidebar.caption("Machine Learning Project")
page = st.sidebar.radio("Navigation", ["🏠 Dashboard", "💰 Price Prediction",
                                       "📊 Data Visualization", "🧠 Model Analysis"])
st.sidebar.divider()
uploaded = st.sidebar.file_uploader("Upload your housing CSV", type=["csv"])

try:
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        df = load_data(Path(__file__).parent / "housing.csv")
except FileNotFoundError:
    st.title("🏡 Boston Housing Price Prediction")
    st.warning("Put `housing.csv` next to `app.py`, or upload it from the sidebar.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"The dataset must contain a {TARGET} column.")
    st.stop()

features = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
if not features:
    st.error("No numeric feature columns were found.")
    st.stop()

data = df[features + [TARGET]].dropna().copy()
if len(data) < 20:
    st.error("Not enough complete rows to train the model.")
    st.stop()

with st.spinner("Training the Decision Tree model..."):
    res = train_model(data, features)
model = res["model"]
best_depth = model.get_params()["max_depth"]
prices = data[TARGET]

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit, Pandas, Plotly and Scikit-learn")

# -------------------------------------------------------------- dashboard --
if page == "🏠 Dashboard":
    st.markdown('<p class="main-title">🏡 Boston Housing Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Estimating home values with a Decision Tree Regressor, '
                'tuned with grid search and cross-validation.</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Dataset Overview")
    c = st.columns(4)
    c[0].metric("Total Records", f"{len(df):,}")
    c[1].metric("Features", len(features))
    c[2].metric("Best Tree Depth", best_depth)
    c[3].metric("Test R² Score", f"{res['r2']:.3f}")

    st.divider()
    st.subheader("Housing Price Statistics")
    c = st.columns(5)
    c[0].metric("Minimum", money(prices.min()))
    c[1].metric("Maximum", money(prices.max()))
    c[2].metric("Mean", money(prices.mean()))
    c[3].metric("Median", money(prices.median()))
    c[4].metric("Std. Deviation", money(prices.std(ddof=0)))

    st.divider()
    st.subheader("Features")
    for f in features:
        st.markdown(f"- **{f}** — {FEATURE_INFO.get(f, 'Numeric feature')}")
    st.markdown(f"- **{TARGET}** — Median value of homes (the target we predict)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Target Distribution")
    st.plotly_chart(px.histogram(data, x=TARGET, nbins=30, title="Distribution of House Prices",
                                 labels={TARGET: "House Price"}), use_container_width=True)

# ------------------------------------------------------------- prediction --
elif page == "💰 Price Prediction":
    st.title("💰 House Price Prediction")
    st.write("Enter the neighborhood characteristics to estimate a home's selling price.")
    st.divider()

    use_clients = set(features) == set(CLIENTS["Client 1"])
    options = ["Custom"] + (list(CLIENTS) if use_clients else [])
    preset = st.radio("Start from", options, horizontal=True)

    st.subheader("House Features")
    cols = st.columns(min(len(features), 3))
    values = {}
    for i, f in enumerate(features):
        lo, hi = float(data[f].min()), float(data[f].max())
        default = float(CLIENTS[preset][f]) if preset in CLIENTS else float(data[f].median())
        default = min(max(default, lo), hi)
        with cols[i % len(cols)]:
            values[f] = st.number_input(f, min_value=lo, max_value=hi, value=default,
                                        step=0.1, format="%.2f", key=f"{f}_{preset}",
                                        help=f"{FEATURE_INFO.get(f, '')} (observed range {lo:g} – {hi:g})")

    input_df = pd.DataFrame([values], columns=features)
    pred = float(model.predict(input_df)[0])

    st.divider()
    st.markdown(f"""<div class="prediction-card"><h3>Estimated House Price</h3>
                <h1>{money(pred)}</h1></div>""", unsafe_allow_html=True)
    st.caption("A machine learning estimate, not a professional property valuation.")

    c = st.columns(3)
    c[0].metric("vs. dataset mean", f"{pred - prices.mean():+,.0f}")
    c[1].metric("vs. dataset median", f"{pred - prices.median():+,.0f}")
    c[2].metric("Percentile in dataset", f"{(prices < pred).mean() * 100:.0f}%")

    fig = px.histogram(data, x=TARGET, nbins=30, opacity=0.7, title="Where this price falls")
    fig.add_vline(x=pred, line_color="red", line_width=3, annotation_text="prediction")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How stable is this prediction? (10 random train/test splits)"):
        if st.button("Run 10 trials"):
            with st.spinner("Refitting the model on 10 different splits..."):
                trials = prediction_trials(data[features], data[TARGET],
                                           tuple(values[f] for f in features), features)
            st.dataframe(pd.DataFrame({"Trial": range(1, 11), "Predicted price": [money(p) for p in trials]}),
                         use_container_width=True, hide_index=True)
            st.metric("Range in prices", money(max(trials) - min(trials)))

# ---------------------------------------------------------- visualization --
elif page == "📊 Data Visualization":
    st.title("📊 Data Visualization")
    t1, t2, t3, t4 = st.tabs(["Overview", "Distributions", "Relationships", "Correlation"])

    with t1:
        c1, c2 = st.columns(2)
        c1.write("Data types")
        c1.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)
        c2.write("Missing values")
        c2.dataframe(df.isnull().sum().rename("Missing"), use_container_width=True)
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

    with t2:
        sel = st.selectbox("Select a feature", features + [TARGET])
        st.plotly_chart(px.histogram(data, x=sel, marginal="box", nbins=30,
                                     title=f"Distribution of {sel}"), use_container_width=True)

    with t3:
        c1, c2 = st.columns(2)
        x_f = c1.selectbox("X-axis", features)
        y_f = c2.selectbox("Y-axis", [TARGET] + features)
        fig = px.scatter(data, x=x_f, y=y_f, opacity=0.6, title=f"{y_f} vs {x_f}")
        if x_f != y_f:
            slope, intercept = np.polyfit(data[x_f], data[y_f], 1)
            xs = np.array([data[x_f].min(), data[x_f].max()])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                     name="Linear trend", line=dict(color="red", dash="dash")))
            st.caption(f"Correlation: **{data[x_f].corr(data[y_f]):.2f}**")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        corr = data.corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                                  color_continuous_scale="RdBu_r", title="Correlation Heatmap"),
                        use_container_width=True)

# --------------------------------------------------------------- analysis --
else:
    st.title("🧠 Model Analysis")
    st.write("Decision Tree Regressor tuned with GridSearchCV (max_depth 1–10, 10 shuffled CV splits).")
    st.divider()

    c = st.columns(4)
    c[0].metric("Test R²", f"{res['r2']:.3f}")
    c[1].metric("MAE", money(res["mae"]))
    c[2].metric("RMSE", money(res["rmse"]))
    c[3].metric("Optimal Depth", best_depth)
    st.caption(f"Mean cross-validation R² at the optimal depth: {res['cv_score']:.3f}")
    st.divider()

    t1, t2, t3, t4 = st.tabs(["Learning Curves", "Complexity Curve",
                              "Actual vs Predicted", "Feature Importance"])

    with t1:
        st.write("Training vs. validation R² as the training set grows.")
        depths = st.multiselect("max_depth values", list(range(1, 11)), default=[1, 3, 6, 10])
        cols = st.columns(2)
        for i, d in enumerate(depths):
            sizes, tr, va = learning_curves(data[features], data[TARGET], d)
            fig = go.Figure()
            add_band(fig, sizes, tr, "Training score", RED)
            add_band(fig, sizes, va, "Validation score", GREEN)
            fig.update_layout(title=f"max_depth = {d}", xaxis_title="Training points",
                              yaxis_title="R² score", yaxis_range=[-0.05, 1.05],
                              legend=dict(orientation="h", y=-0.25))
            cols[i % 2].plotly_chart(fig, use_container_width=True)

    with t2:
        d, tr, va = complexity_curve(res["X_train"], res["y_train"])
        fig = go.Figure()
        add_band(fig, d, tr, "Training score", RED)
        add_band(fig, d, va, "Validation score", GREEN)
        fig.add_vline(x=best_depth, line_dash="dot", annotation_text=f"optimal = {best_depth}")
        fig.update_layout(title="Decision Tree Regressor Complexity Performance",
                          xaxis_title="Maximum depth", yaxis_title="R² score",
                          yaxis_range=[-0.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)
        st.info("Low depth underfits (high bias); high depth overfits (high variance). "
                "A big gap between the two curves signals overfitting.")

    with t3:
        comp = pd.DataFrame({"Actual Price": res["y_test"].to_numpy(), "Predicted Price": res["y_pred"]})
        fig = px.scatter(comp, x="Actual Price", y="Predicted Price", opacity=0.7, title="Actual vs Predicted")
        lo, hi = comp.min().min(), comp.max().max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                                 line=dict(dash="dash", color="red")))
        st.plotly_chart(fig, use_container_width=True)
        st.write("Points closer to the dashed line are more accurate predictions.")

    with t4:
        imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}) \
            .sort_values("Importance")
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                               title="Decision Tree Feature Importance"), use_container_width=True)