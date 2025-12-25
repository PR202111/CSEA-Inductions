import streamlit as st
import requests


API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Green/Red Flag Detector", page_icon="💚🟥")

st.title("AI Green/Red Flag Detector")
st.write("Analyze dating bios for green, yellow, and red flags, and optionally get a rewritten version.")


bio_input = st.text_area("Enter a dating bio:", height=150)



if st.button("Analyze Bio"):
    if not bio_input.strip():
        st.warning("Please enter a bio to analyze.")
    else:
        try:
            classify_resp = requests.post(
                f"{API_URL}/classify",
                json={"text": bio_input}
            ).json()
        except Exception as e:
            st.error(f"Error calling backend API: {e}")
            classify_resp = None

        if classify_resp:
            st.subheader("⚡ Flag Analysis")
            flags = classify_resp.get("flags", {})
            scores = classify_resp.get("scores", {})


            st.markdown("**Green Flags:**")
            for f in flags.get("green_flags", []):
                st.write(f"- {f['reason']}: {f['phrases']}")
            if not flags.get("green_flags"):
                st.write("- None")

            st.markdown("**Yellow Flags:**")
            for f in flags.get("yellow_flags", []):
                st.write(f"- {f['reason']}: {f['phrases']}")
            if not flags.get("yellow_flags"):
                st.write("- None")

            st.markdown("**Red Flags:**")
            for f in flags.get("red_flags", []):
                st.write(f"- {f['reason']}: {f['phrases']}")
            if not flags.get("red_flags"):
                st.write("- None")

            st.subheader("Model Scores")
            for k, v in scores.items():
                st.write(f"- {k}: {v:.2f}")


target_tone = st.selectbox(
    "Select target tone for rewritten bio (only used in Rewrite):",
    ["healthy & respectful", "confident but humble", "playful & witty", "emotionally mature", "calm & grounded", "light-hearted"]
)


if st.button("Rewrite Bio"):
    if not bio_input.strip():
        st.warning("Please enter a bio to analyze and rewrite.")
    else:

        try:
            rewrite_resp = requests.post(
                f"{API_URL}/rewrite",
                json={"text": bio_input, "target_tone": target_tone}
            ).json()
        except Exception as e:
            st.error(f"Error calling rewrite API: {e}")
            rewrite_resp = None

        if rewrite_resp:
            st.subheader("Improved Bio")
            st.write(rewrite_resp.get("improved_bio", ""))

            
