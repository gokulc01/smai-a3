import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="🍛 Indian Food Classifier",
    page_icon="🍛",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================================
# Food Database with Calories & Allergens
# ==========================================
FOOD_DATABASE = {
    "bhature": {
        "calories": 450,
        "description": "Fried Indian bread made with yogurt, often served with chickpeas",
        "allergens": ["wheat", "dairy", "gluten"],
    },
    "chapati": {
        "calories": 100,
        "description": "Unleavened Indian flatbread",
        "allergens": ["wheat", "gluten"],
    },
    "daal_baati_churma": {
        "calories": 800,
        "description": "Baked bread served with lentil curry and sweet crumbled mixture",
        "allergens": ["wheat", "dairy", "gluten", "legumes"],
    },
    "lachha_paratha": {
        "calories": 350,
        "description": "Layered Indian flatbread with ghee, flaky and crispy",
        "allergens": ["wheat", "dairy", "gluten"],
    },
    "naan": {
        "calories": 270,
        "description": "Leavened Indian flatbread made with yogurt and butter",
        "allergens": ["wheat", "dairy", "gluten", "yeast"],
    },
    "palak_roti": {
        "calories": 120,
        "description": "Indian flatbread made with spinach and whole wheat flour",
        "allergens": ["wheat", "gluten"],
    },
    "ragi_roti": {
        "calories": 110,
        "description": "Nutritious, naturally gluten-free flatbread made from finger millet flour",
        "allergens": [],
    },
    "rumali_rotti": {
        "calories": 160,
        "description": "Thin, delicate Indian rolled flatbread, similar to a crepe",
        "allergens": ["wheat", "gluten"],
    },
}

# Allergen colors (RGB)
ALLERGEN_COLORS = {
    "wheat": "#FFD700",  # Gold
    "dairy": "#87CEEB",  # Sky Blue
    "gluten": "#FF6347",  # Tomato Red
    "legumes": "#90EE90",  # Light Green
    "corn": "#FFB347",  # Peach
    "ghee": "#DEB887",  # Burlywood
    "yeast": "#DDA0DD",  # Plum
}


# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    """Load the trained MobileNetV2 model"""
    try:
        # Update the filename to your most recent fine-tuned model
        model = tf.keras.models.load_model("indian_food_classifier_finetuned.keras")
        return model
    except FileNotFoundError:
        st.error(
            "❌ Model file not found! Please ensure 'indian_food_classifier_finetuned.keras' is in the directory."
        )
        return None


# ==========================================
# Prediction Function
# ==========================================
def predict_food(image, model, class_names, img_size=(224, 224)):
    """Predict the food class and return probabilities"""
    try:
        # Preprocess the image
        img = image.resize(img_size)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)

        # Get class probabilities
        pred_dict = {
            class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
        }
        pred_dict = dict(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True))

        return pred_dict
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# ==========================================
# Main App
# ==========================================
def main():
    st.title("🍛 Indian Food Classifier")
    st.markdown("Upload a food photo to identify the dish and view nutritional info")
    st.divider()

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Get class names
    class_names = sorted(list(FOOD_DATABASE.keys()))

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "This app uses a trained MobileNetV2 model to identify Indian dishes and provide:"
        )
        st.write("✓ Predicted dish name")
        st.write("✓ Approximate calories")
        st.write("✓ Common allergens")
        st.write("✓ Confidence score")

        st.divider()
        st.header("📊 Supported Dishes")
        for dish in class_names:
            st.write(f"• {dish.replace('_', ' ').title()}")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Upload Food Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of an Indian dish",
        )

    with col2:
        if uploaded_file is not None:
            st.subheader("Preview")
            image = Image.open(uploaded_file)
            # Display image, forcing RGB conversion to avoid issues with some PNGs
            st.image(image.convert("RGB"), use_container_width=True)

    # Prediction section
    if uploaded_file is not None:
        st.divider()

        with st.spinner("🔍 Analyzing image..."):
            predictions = predict_food(image.convert("RGB"), model, class_names)

        if predictions is not None:
            # Get top prediction
            top_dish = list(predictions.keys())[0]
            confidence = predictions[top_dish] * 100

            # Display main prediction
            st.subheader("🎯 Prediction Result")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    label="Predicted Dish",
                    value=top_dish.replace("_", " ").title(),
                    delta=f"Confidence: {confidence:.1f}%",
                )

            st.divider()

            # Display food info
            if top_dish in FOOD_DATABASE:
                food_info = FOOD_DATABASE[top_dish]

                # Calories
                st.subheader("🔥 Nutritional Information")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.metric(
                        label="Calories (approx)", value=f"{food_info['calories']} kcal"
                    )

                with col2:
                    st.write("")  # Spacer
                    st.write("")
                    st.info(food_info["description"])

                # Allergens
                st.subheader("⚠️ Allergen Information")

                if len(food_info["allergens"]) > 0:
                    st.write(
                        f"**Common allergens in {top_dish.replace('_', ' ').title()}:**"
                    )
                    # Display allergens as columns for better layout
                    allergen_cols = st.columns(len(food_info["allergens"]))
                    for idx, allergen in enumerate(food_info["allergens"]):
                        with allergen_cols[idx]:
                            color = ALLERGEN_COLORS.get(allergen, "#CCCCCC")
                            st.markdown(
                                f"<p style='background-color: {color}; color: black; padding: 8px 12px; border-radius: 8px; text-align: center; font-weight: bold;'>{allergen.upper()}</p>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.success(
                        f"**{top_dish.replace('_', ' ').title()} is generally free of common major allergens.**"
                    )

                st.write("")
                # Confidence breakdown
                st.subheader("📊 Model Confidence Breakdown")

                # Create bar chart
                top_5 = dict(list(predictions.items())[:5])
                chart_data = {
                    dish.replace("_", " ").title(): conf * 100
                    for dish, conf in top_5.items()
                }

                st.bar_chart(chart_data)

            st.divider()
            st.success("✅ Prediction complete!")

    else:
        st.info("👆 Upload an image to get started!")


if __name__ == "__main__":
    main()
