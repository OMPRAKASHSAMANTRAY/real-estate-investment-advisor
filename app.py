
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import urllib.request

# --------------------------------------------------
# 1. PROJECT PATHS AND MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CLASSIFIER_PATH = BASE_DIR / "models" / "investment_classifier.joblib"
REGRESSOR_PATH = BASE_DIR / "models" / "five_year_price_regressor.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "real_estate_ml_dataset.csv"

HF_BASE_URL = (
    "https://huggingface.co/omprakash2712/"
    "real-estate-investment-assets/resolve/main/"
)


st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
)


@st.cache_data
def ensure_asset_download(url: str, destination: str) -> str:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if not destination_path.exists():
        urllib.request.urlretrieve(url, destination_path)

    return str(destination_path)



@st.cache_resource
def load_models():
    classifier_path = CLASSIFIER_PATH

    if not classifier_path.exists():
        classifier_path = Path(
            ensure_asset_download(
                HF_BASE_URL + "investment_classifier.joblib",
                str(CLASSIFIER_PATH),
            )
        )

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(REGRESSOR_PATH)
    return classifier, regressor



@st.cache_data
def load_reference_data():
    data_path = DATA_PATH

    if not data_path.exists():
        data_path = Path(
            ensure_asset_download(
                HF_BASE_URL + "real_estate_ml_dataset.csv",
                str(DATA_PATH),
            )
        )

    return pd.read_csv(data_path)


try:
    classifier, regressor = load_models()
    reference_df = load_reference_data()
except Exception as exc:
    st.error(f"Unable to load the models or dataset: {exc}")
    st.stop()

# --------------------------------------------------
# 2. APP HEADER
# --------------------------------------------------

# PROFESSIONAL APP STYLING
st.markdown("""
<style>
    :root {
        --emerald: #10b981;
        --emerald-light: #6ee7b7;
        --forest: #0b1713;
        --panel: #12221c;
        --panel-light: #192d25;
        --muted: #9aafa4;
        --border: rgba(110, 231, 183, 0.16);
    }

    
    
    
    .stApp {
        background: linear-gradient(
            -45deg,
            #061510,
            #103b2c,
            #0a241c,
            #14543b,
            #071711
        );
        background-size: 400% 400%;
        animation: emeraldFlow 14s ease infinite;
        color: #edf7f1;
    }

    @keyframes emeraldFlow {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }


    @keyframes emeraldFlow {
        0% {
            background-position: 0% 0%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 100%;
        }
    }

    
    /* Subtle architectural line pattern */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0.18;
        background-image:
            linear-gradient(
                90deg,
                transparent 49.8%,
                rgba(167, 243, 208, 0.16) 50%,
                transparent 50.2%
            ),
            linear-gradient(
                0deg,
                transparent 49.8%,
                rgba(167, 243, 208, 0.12) 50%,
                transparent 50.2%
            );
        background-size: 110px 110px;
        animation: architectureDrift 35s linear infinite;
    }

    @keyframes architectureDrift {
        from {
            background-position: 0 0;
        }
        to {
            background-position: 90px 90px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp,
        .stApp::before {
            animation: none;
        }
    }


    .block-container {
        max-width: 1500px;
        padding: 2rem 2.5rem 3rem;
    }

    h1, h2, h3 {
        color: #f0fdf4;
        letter-spacing: -0.5px;
    }

    h1 {
        font-weight: 750;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        line-height: 1.65;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1713, #12251c);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: #e4f5eb;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(25, 45, 37, 0.96),
            rgba(16, 33, 26, 0.94)
        );
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 18px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.14);
        transition: transform 180ms ease, border-color 180ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(110, 231, 183, 0.48);
    }

    div[data-testid="stMetricLabel"] {
        color: #a9c2b4;
    }

    div[data-testid="stMetricValue"] {
        color: #6ee7b7;
        font-weight: 750;
    }

    div[data-testid="stForm"],
    div[data-testid="stExpander"] {
        background: rgba(18, 34, 28, 0.72);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: #04150e;
        border: none;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 750;
        transition: transform 180ms ease, filter 180ms ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.12);
        transform: translateY(-2px);
        color: #04150e;
        border: none;
    }

    input, textarea {
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    hr {
        border-color: var(--border);
        margin: 1.6rem 0;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(120deg, #142b45, #203d5c);
    border: 1px solid #36516e;
    margin-bottom: 22px;
">
    <div style="
        color: #9fc5f8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    ">
        PROPERTY ANALYTICS PLATFORM
    </div>
    <h1 style="
        color: white;
        margin: 12px 0 8px 0;
        font-size: 38px;
    ">
        🏡 Real Estate Investment Advisor
    </h1>
    <p style="
        color: #d5e2f0;
        font-size: 16px;
        margin-bottom: 0;
    ">
        Analyze property affordability, compare market indicators,
        and explore an illustrative five-year price scenario.
    </p>
</div>
""", unsafe_allow_html=True)




st.info(
    "Important: the classification represents a median-price-based "
    "label from the training dataset. The five-year projection assumes "
    "8% annual appreciation. Neither result guarantees investment returns."
)

# --------------------------------------------------
# 3. REFERENCE OPTIONS
# --------------------------------------------------

def choices(column, fallback):
    if column in reference_df.columns:
        values = (
            reference_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        return sorted(values) if values else fallback
    return fallback


states = sorted(reference_df["State"].dropna().unique().tolist())
cities = sorted(reference_df["City"].dropna().unique().tolist())
localities = sorted(reference_df["Locality"].dropna().unique().tolist())

property_types = choices(
    "Property_Type",
    ["Apartment", "Independent House", "Villa"],
)
furnishing_options = choices(
    "Furnished_Status",
    ["Unfurnished", "Semi-furnished", "Furnished"],
)
transport_options = choices(
    "Public_Transport_Accessibility",
    ["Low", "Medium", "High"],
)
amenity_options = choices(
    "Amenities",
    ["Garden", "Pool", "Gym", "Clubhouse", "Playground"],
)
facing_options = choices("Facing", ["North", "South", "East", "West"])
owner_options = choices("Owner_Type", ["Owner", "Builder", "Broker"])
availability_options = choices(
    "Availability_Status",
    ["Ready_to_Move", "Under_Construction"],
)
security_options = choices("Security", ["Yes", "No"])
parking_options = choices("Parking_Space", ["Yes", "No"])

# --------------------------------------------------
# 4. PROPERTY INPUT FORM
# --------------------------------------------------

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("## 🏡 Property Advisor")
    st.caption("PREMIUM ANALYTICS")

    st.markdown("---")

    selected_section = st.radio(
        "Go to section",
        [
            "📋 Property Details",
            "📊 Investment Analysis",
            "📈 Five-Year Scenario",
            "🧾 Property Summary",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("### About this dashboard")
    st.caption(
        "Analyze property details, review model outputs, "
        "and explore an illustrative five-year price scenario."
    )

    st.info(
        "The five-year scenario assumes 8% annual appreciation. "
        "Actual property prices may differ."
    )

    st.markdown(
    '<div id="property-details"></div>',
    unsafe_allow_html=True,
)

st.header("1. Enter Property Details")

with st.form("property_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
            state = st.selectbox("State", states)

            cities_for_state = sorted(
                reference_df.loc[
                    reference_df["State"] == state, "City"
                ].dropna().unique().tolist()
            )

            city = st.selectbox("City", cities_for_state)

            localities_for_city = sorted(
                reference_df.loc[
                    (reference_df["State"] == state)
                    & (reference_df["City"] == city),
                    "Locality"
                ].dropna().unique().tolist()
            )

            locality = st.selectbox("Locality", localities_for_city)

            property_type = st.selectbox("Property Type", property_types)
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])


    with col2:
        size = st.number_input(
            "Property Size (sq. ft.)",
            min_value=500,
            max_value=5000,
            value=1500,
            step=50,
        )
        price = st.number_input(
            "Current Property Price (₹ lakhs)",
            min_value=10.0,
            max_value=500.0,
            value=150.0,
            step=5.0,
        )
        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=2026,
            value=2015,
            step=1,
        )
        floor_no = st.number_input(
            "Floor Number",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
        )
        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )

    with col3:
        furnished = st.selectbox(
            "Furnishing Status", furnishing_options
        )
        transport = st.selectbox(
            "Public Transport Accessibility", transport_options
        )
        parking = st.selectbox("Parking Available", parking_options)
        security = st.selectbox("Security Available", security_options)
        facing = st.selectbox("Facing Direction", facing_options)
        owner_type = st.selectbox("Ownership Type", owner_options)
        availability = st.selectbox(
            "Availability Status", availability_options
        )
        amenities = st.multiselect(
            "Amenities",
            amenity_options,
            default=amenity_options[:1],
        )

    schools = st.slider("Nearby Schools", 1, 10, 5)
    hospitals = st.slider("Nearby Hospitals", 1, 10, 5)

    submitted = st.form_submit_button(
        "Analyze Property",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# 5. PREDICTION LOGIC
# --------------------------------------------------

if submitted:
    if size <= 0 or price <= 0:
        st.error("Property size and price must be positive.")
        st.stop()

    if floor_no > total_floors:
        st.warning(
            "The selected floor exceeds the total floors. "
            "Check the property details before proceeding."
        )
        st.stop()

    # These definitions must match the training pipeline.
    # The project uses 2025 as the reference year in its audit.
    reference_year = 2025
    age = reference_year - int(year_built)

    # Price in lakhs / area in square feet = lakhs per sq. ft.
    price_per_sqft_lakhs = float(price) / float(size)

    # Calculated price per square foot in rupees.
    calculated_price_per_sqft_rupees = price_per_sqft_lakhs * 100000

    # --------------------------------------------------
    # CLASSIFICATION FEATURES
    # --------------------------------------------------

    classifier_features = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": int(bhk),
        "Size_in_SqFt": float(size),
        "Year_Built": int(year_built),
        "Furnished_Status": furnished,
        "Floor_No": int(floor_no),
        "Total_Floors": int(total_floors),
        "Age_of_Property": int(age),
        "Nearby_Schools": int(schools),
        "Nearby_Hospitals": int(hospitals),
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": ", ".join(amenities),
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Floor_Data_Quality_Issue": int(floor_no > total_floors),
        "Calculated_Price_Per_SqFt_Lakhs": price_per_sqft_lakhs,
    }

    # --------------------------------------------------
    # REGRESSION FEATURES
    # --------------------------------------------------

    regression_features = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": int(bhk),
        "Size_in_SqFt": float(size),
        "Price_in_Lakhs": float(price),
        "Price_per_SqFt": price_per_sqft_lakhs,
        "Year_Built": int(year_built),
        "Furnished_Status": furnished,
        "Floor_No": int(floor_no),
        "Total_Floors": int(total_floors),
        "Age_of_Property": int(age),
        "Nearby_Schools": int(schools),
        "Nearby_Hospitals": int(hospitals),
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": ", ".join(amenities),
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Floor_Data_Quality_Issue": int(floor_no > total_floors),
        "Calculated_Price_Per_SqFt_Lakhs": price_per_sqft_lakhs,
    }

    classifier_input = pd.DataFrame(
        [classifier_features],
        columns=list(classifier.feature_names_in_),
    )

    regressor_input = pd.DataFrame(
        [regression_features],
        columns=list(regressor.feature_names_in_),
    )

    try:
        predicted_label = int(classifier.predict(classifier_input)[0])
        future_price_model = float(
            regressor.predict(regressor_input)[0]
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # --------------------------------------------------
    # 6. RESULTS
    # --------------------------------------------------

    st.header("2. Property Analysis")
    reference_year = 2025
    age = reference_year - int(year_built)

    price_per_sqft_lakhs = float(price) / float(size)
    calculated_price_per_sqft_rupees = price_per_sqft_lakhs * 100000

    formula_price = float(price) * (1 + 0.08) ** 5
    
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    with result_col1:
        st.metric(
            "🏠 Current Property Price",
            f"₹{price:,.2f} lakh",
            help="Current property price entered in the form."
        )

    with result_col2:
        st.metric(
            "📐 Price per Sq. Ft.",
            f"₹{calculated_price_per_sqft_rupees:,.0f}",
            help="Calculated from property price and property size."
        )
        

    with result_col3:
        st.metric(
            "🗓️ Property Age",
            f"{age} years",
            help="Calculated using the project's 2025 reference year."
        )

    
    with result_col4:
        st.metric(
            "📈 Five-Year Scenario",
            f"₹{formula_price:,.2f} lakh",
            delta="Illustrative estimate",
            help="Assumes 8% annual appreciation; not a market forecast."
        )


        st.markdown(
    '<div id="investment-analysis"></div>',
    unsafe_allow_html=True,
    )

    
    st.subheader("3. Affordability Classification")

    if predicted_label == 1:
        st.success("### Good Investment\n\nThe model assigns this property to the dataset's lower-price category.")
    else:
        st.warning("### Not Good Investment\n\nThe model assigns this property to the dataset's higher-price category.")

    st.caption(
        "This is a median-price-based classification, not a verified "
        "assessment of profitability or investment risk."
    )


    # --------------------------------------------------
    # 7. FIVE-YEAR SCENARIO
    # --------------------------------------------------

    st.markdown(
    '<div id="five-year-scenario"></div>',
    unsafe_allow_html=True,
    )

    st.header("3. Five-Year Price Scenario")

    annual_appreciation = 0.08
    years = np.arange(0, 6)
    scenario_prices = [
        price * (1 + annual_appreciation) ** year
        for year in years
    ]

    formula_price = scenario_prices[-1]

    st.metric(
        "Scenario Price After 5 Years",
        f"₹{formula_price:,.2f} lakh",
        delta=f"{((formula_price / price) - 1) * 100:.2f}% over 5 years",
    )

    st.write(
        f"The saved regression model estimates ₹{future_price_model:,.2f} lakh. "
        f"The direct 8% formula gives ₹{formula_price:,.2f} lakh."
    )

    st.caption(
        "The formula is an illustrative scenario, not a market forecast. "
        "The regression model was trained on this formula-generated target, "
        "so its output is not independent evidence of future appreciation."
    )

    
    # FIVE-YEAR PRICE CHART
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(
        years,
        scenario_prices,
        marker="o",
        linewidth=2.5,
        markersize=7,
    )

    ax.fill_between(years, scenario_prices, alpha=0.12)

    ax.set_title(
        "Illustrative Five-Year Property Price Scenario",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Years from Today")
    ax.set_ylabel("Estimated Price (₹ lakhs)")
    ax.set_xticks(years)
    ax.grid(True, linestyle="--", alpha=0.3)

    for x, y in zip(years, scenario_prices):
        ax.annotate(
            f"₹{y:.2f}L",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    # --------------------------------------------------
    # 8. PROPERTY SUMMARY
    # --------------------------------------------------

    st.markdown(
    '<div id="property-summary"></div>',
    unsafe_allow_html=True,
    )

    st.header("4. Property Summary")

    summary = pd.DataFrame(
        list(classifier_features.items()),
        columns=["Feature", "Value"],
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Property Analysis",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="property_analysis.csv",
        mime="text/csv",
    )
