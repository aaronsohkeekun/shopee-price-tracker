# =============================================================================
# SHOPEE PRICE TRACKER - app.py  (v5 — CSV storage, ready for Streamlit Cloud)
# =============================================================================
# STORAGE: Local CSV file (price_history.csv)
# NOTE: On Streamlit Community Cloud, this CSV resets if the app restarts.
#       Use the Download button regularly to back up your data locally.
#       When ready, upgrade to Google Sheets for permanent cloud storage.
#
# FEATURES:
#   • Paste screenshot from clipboard (Mac: Cmd+Ctrl+Shift+4)
#   • Upload screenshot(s) from file (works on all devices)
#   • Upload multiple screenshots — AI reads all together
#   • Manual override fields for Title, Price, Seller if AI misses them
#   • Stock/Quantity is optional — saves as N/A if not found
#   • Search + dropdown to find products in the dashboard
#   • Download full history as CSV anytime
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st               # Builds the entire web page
import pandas as pd                  # Works with table/spreadsheet data
import os                            # Checks whether files exist on disk
import json                          # Reads structured data returned by the AI
import re                            # Finds patterns in text (cleans AI output)
from datetime import datetime        # Gets the current date and time
from PIL import Image                # Opens and inspects image files
import io                            # Reads image data held in memory

# New official Google AI SDK
from google import genai
from google.genai import types


# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_FILE = "price_history.csv"

CSV_COLUMNS = [
    "timestamp",        # Date + time the entry was saved
    "tracking_id",      # "Product Title | Seller Name"
    "product_title",    # The product's full name
    "seller_name",      # The shop/seller name
    "price",            # Current price (e.g. "RM 25.90")
    "quantity_left",    # Stock remaining — "N/A" if not found
    "product_url",      # Shopee URL (for reference only)
    "num_screenshots",  # How many screenshots were uploaded
    "field_sources",    # Which fields came from AI vs manual input
]

GEMINI_MODEL = "gemini-2.5-flash-lite"

MISSING_VALUES = {
    "not found", "n/a", "", "none", "unknown",
    "not available", "not visible", "unavailable"
}


# =============================================================================
# PAGE SETUP
# =============================================================================

st.set_page_config(
    page_title="Shopee Price Tracker",
    page_icon="🛒",
    layout="wide",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gemini_client():
    """
    Creates a Gemini API client using the secret API key stored in
    Streamlit Secrets. Returns the client on success, None on failure.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client  = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error(
            "🔑 **API Key Not Found!**\n\n"
            "Add this to your Streamlit Secrets:\n"
            "```\nGEMINI_API_KEY = \"your-key-here\"\n```\n\n"
            "**Locally:** edit `.streamlit/secrets.toml`\n"
            "**Streamlit Cloud:** App Settings → Secrets"
        )
        return None
    except Exception as e:
        st.error(f"❌ Error connecting to Gemini: {e}")
        return None


def load_price_history():
    """
    Reads the CSV file and returns it as a pandas DataFrame.
    If the file doesn't exist yet, returns an empty table.
    """
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = "N/A"
        return df
    return pd.DataFrame(columns=CSV_COLUMNS)


def save_new_entry(record: dict):
    """
    Appends one new row to the CSV file.
    Never deletes existing rows — history grows forever downward.
    """
    new_df      = pd.DataFrame([record])
    file_exists = os.path.exists(CSV_FILE)
    new_df.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)


def build_image_part(image_bytes: bytes):
    """
    Converts raw image bytes into a types.Part object the Gemini SDK can send.
    Detects the correct MIME type (jpeg, png, webp, etc.) automatically.
    """
    pil_img  = Image.open(io.BytesIO(image_bytes))
    fmt      = pil_img.format if pil_img.format else "JPEG"
    mime_map = {
        "JPEG": "image/jpeg",
        "PNG":  "image/png",
        "WEBP": "image/webp",
        "GIF":  "image/gif",
    }
    return types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_map.get(fmt, "image/jpeg")
    )


def extract_data_with_gemini(client, all_image_bytes: list) -> tuple:
    """
    Sends ALL uploaded screenshots to Gemini AI in one single request.
    Returns (result_dict, None) on success or (None, error_string) on failure.
    """
    num_images    = len(all_image_bytes)
    image_context = (
        "You are given 1 screenshot of a Shopee product page."
        if num_images == 1 else
        f"You are given {num_images} screenshots of the SAME Shopee product page, "
        f"taken from different scroll positions. Treat them as one complete page."
    )

    prompt = f"""
You are a data extraction assistant. {image_context}

Using ALL the images together, extract these four pieces of information:

1. product_title  — The full product name/title.
2. price          — Current selling price with currency symbol (e.g. "RM 25.90").
                    Use the discounted price if two prices are shown.
                    If a range (e.g. RM10-RM20), return the lower value.
3. seller_name    — Shop/seller name. Usually near "Chat Now" or "Visit Shop".
4. quantity_left  — Remaining stock (e.g. "47"). Return "N/A" if not visible.

Return ONLY a valid JSON object with exactly these four keys.
No explanation, no markdown, no code fences — just raw JSON:
{{
  "product_title": "...",
  "price": "...",
  "seller_name": "...",
  "quantity_left": "..."
}}
"""

    try:
        contents = [build_image_part(b) for b in all_image_bytes] + [prompt]
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )
        raw_text = response.text.strip()

        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            return None, f"AI did not return valid JSON.\nRaw response:\n{raw_text}"

        data = json.loads(match.group())
        return {
            "product_title": data.get("product_title", "Not Found"),
            "price":         data.get("price",         "Not Found"),
            "seller_name":   data.get("seller_name",   "Not Found"),
            "quantity_left": data.get("quantity_left", "N/A"),
        }, None

    except json.JSONDecodeError as e:
        return None, f"Could not parse AI response as JSON: {e}"
    except Exception as e:
        return None, f"Gemini API error: {e}"


def is_missing(value: str) -> bool:
    """Returns True if a value is blank or a known 'not found' placeholder."""
    if not value:
        return True
    return str(value).strip().lower() in MISSING_VALUES


def create_tracking_id(title: str, seller: str) -> str:
    """Builds the unique tracking key: 'Product Title | Seller Name'"""
    return f"{title.strip()} | {seller.strip()}"


# =============================================================================
# MAIN APP
# =============================================================================

def main():

    st.title("🛒 Shopee Price Tracker")
    st.markdown(
        "Upload **one or more** screenshots of a Shopee product page. "
        "The AI reads all of them together and extracts the price data. "
        "Any field the AI misses can be filled in manually before saving."
    )
    st.divider()

    # Connect to Gemini — stop if connection fails
    client = get_gemini_client()
    if client is None:
        st.stop()

    # Load price history from CSV
    price_history_df = load_price_history()


    # =========================================================================
    # SECTION A — UPLOAD & EXTRACT
    # =========================================================================
    with st.expander("📸 **Track a New Price Entry**", expanded=True):

        # Step 1 — URL
        st.markdown("#### Step 1 — Product URL *(optional)*")
        product_url = st.text_input(
            label="Paste the Shopee product URL here (saved for your records only)",
            placeholder="https://shopee.com.my/product/...",
        )

        st.divider()

        # Step 2 — Screenshots
        st.markdown("#### Step 2 — Add Screenshot(s)")
        st.markdown(
            "Add screenshots using **either or both** methods below. "
            "All added images are sent to the AI together."
        )

        # ── Method A: Clipboard Paste ──────────────────────────────────────
        st.markdown(
            "**📋 Method A — Paste from Clipboard** "
            "*(Mac: `Cmd+Ctrl+Shift+4` → select area → click button below)*"
        )

        try:
            from streamlit_paste_button import paste_image_button

            paste_result = paste_image_button(
                label="📋 Click here to paste screenshot from clipboard",
                background_color="#2d6a4f",
                hover_background_color="#1b4332",
                key="paste_btn",
            )

            if paste_result and paste_result.image_data is not None:
                paste_buffer = io.BytesIO()
                paste_result.image_data.save(paste_buffer, format="PNG")
                pasted_bytes = paste_buffer.getvalue()

                if "pasted_images" not in st.session_state:
                    st.session_state.pasted_images = []

                if pasted_bytes not in st.session_state.pasted_images:
                    st.session_state.pasted_images.append(pasted_bytes)
                    st.success(
                        f"✅ Screenshot pasted! "
                        f"({len(st.session_state.pasted_images)} pasted image(s) ready)"
                    )

            if "pasted_images" in st.session_state and st.session_state.pasted_images:
                st.markdown(f"*{len(st.session_state.pasted_images)} pasted image(s):*")
                paste_cols = st.columns(min(len(st.session_state.pasted_images), 4))
                for i, pb in enumerate(st.session_state.pasted_images):
                    with paste_cols[i % 4]:
                        st.image(pb, caption=f"Pasted {i + 1}", use_container_width=True)

                if st.button("🗑️ Clear pasted images", key="clear_paste"):
                    st.session_state.pasted_images = []
                    st.rerun()

        except ImportError:
            st.warning(
                "⚠️ Clipboard paste not available. Install it with:\n"
                "```\npip install streamlit-paste-button\n```\n"
                "Then restart the app. File upload below still works."
            )

        st.markdown("---")

        # ── Method B: File Upload ──────────────────────────────────────────
        st.markdown(
            "**📁 Method B — Upload File(s)** *(works on all devices including mobile)*"
        )

        uploaded_files = st.file_uploader(
            label="Upload screenshot(s) — JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="💡 On mobile: select multiple photos from your gallery at once.",
        )

        if uploaded_files:
            st.markdown(f"*{len(uploaded_files)} uploaded file(s):*")
            cols_per_row = min(len(uploaded_files), 4)
            preview_cols = st.columns(cols_per_row)
            for i, uf in enumerate(uploaded_files):
                with preview_cols[i % cols_per_row]:
                    st.image(uf, caption=f"Upload {i + 1}", use_container_width=True)

        st.divider()

        # Step 3 — Manual overrides
        st.markdown("#### Step 3 — Manual Overrides *(fill in if AI misses anything)*")
        st.markdown(
            "Leave blank to let the AI fill automatically. "
            "Type here if AI gets something wrong — **your input always wins.** "
            "Stock is optional and saves as N/A if left blank."
        )

        oc1, oc2 = st.columns(2)
        with oc1:
            manual_title = st.text_input(
                "📦 Product Title",
                placeholder="e.g. Wireless Bluetooth Earbuds Pro Max",
                key="manual_title",
            )
            manual_price = st.text_input(
                "💰 Price",
                placeholder="e.g. RM 25.90",
                key="manual_price",
            )
        with oc2:
            manual_seller = st.text_input(
                "🏪 Seller / Shop Name",
                placeholder="e.g. TechShopMY Official Store",
                key="manual_seller",
            )
            manual_stock = st.text_input(
                "📦 Stock / Quantity Left *(optional)*",
                placeholder="e.g. 47  — or leave blank for N/A",
                key="manual_stock",
            )

        st.divider()

        # Step 4 — Analyze button
        st.markdown("#### Step 4 — Analyze & Save")

        pasted_bytes_list   = st.session_state.get("pasted_images", [])
        uploaded_bytes_list = [uf.read() for uf in uploaded_files] if uploaded_files else []
        total_images        = len(pasted_bytes_list) + len(uploaded_bytes_list)
        no_images           = total_images == 0

        analyze_clicked = st.button(
            label=(
                f"🤖 Analyze {total_images} Screenshot(s) with AI & Save"
                if total_images > 0
                else "🤖 Analyze Screenshot(s) with AI & Save"
            ),
            type="primary",
            use_container_width=True,
            disabled=no_images,
        )
        if no_images:
            st.caption("⬆️ Paste or upload at least one screenshot to enable this button.")

        # --- Processing block ---
        if analyze_clicked and not no_images:

            all_image_bytes = pasted_bytes_list + uploaded_bytes_list

            with st.spinner(
                f"Sending {len(all_image_bytes)} screenshot(s) to Gemini AI "
                f"({GEMINI_MODEL})… Please wait 5–25 seconds…"
            ):
                ai_result, error_msg = extract_data_with_gemini(client, all_image_bytes)

            if error_msg:
                st.error(f"❌ AI extraction failed:\n\n{error_msg}")
                st.info(
                    "💡 **You can still save manually.** "
                    "Fill in Step 3 above and click Analyze again."
                )
                ai_result = {
                    "product_title": "Not Found",
                    "price":         "Not Found",
                    "seller_name":   "Not Found",
                    "quantity_left": "N/A",
                }

            # Merge AI + manual overrides
            def resolve(manual_val, ai_val, fallback="Not Found"):
                mv = manual_val.strip() if manual_val else ""
                if mv:
                    return mv, "Manual"
                elif not is_missing(ai_val):
                    return ai_val, "AI"
                else:
                    return fallback, "fallback"

            final_title,  src_title  = resolve(manual_title,  ai_result["product_title"])
            final_price,  src_price  = resolve(manual_price,  ai_result["price"])
            final_seller, src_seller = resolve(manual_seller, ai_result["seller_name"])
            final_stock,  src_stock  = resolve(manual_stock,  ai_result["quantity_left"], "N/A")

            # Check required fields
            missing_required = []
            if is_missing(final_title):  missing_required.append("Product Title")
            if is_missing(final_price):  missing_required.append("Price")
            if is_missing(final_seller): missing_required.append("Seller Name")

            if missing_required:
                st.warning(
                    "⚠️ **Required field(s) missing:**\n\n"
                    + "\n".join(f"- **{f}**" for f in missing_required)
                    + "\n\nFill them in **Step 3** above and click **Analyze** again."
                )
                st.stop()

            source_summary = (
                f"title={src_title}, price={src_price}, "
                f"seller={src_seller}, stock={src_stock}"
            )

            timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tracking_id = create_tracking_id(final_title, final_seller)

            record = {
                "timestamp":       timestamp,
                "tracking_id":     tracking_id,
                "product_title":   final_title,
                "seller_name":     final_seller,
                "price":           final_price,
                "quantity_left":   final_stock,
                "product_url":     product_url if product_url else "N/A",
                "num_screenshots": len(all_image_bytes),
                "field_sources":   source_summary,
            }
            save_new_entry(record)

            st.success("✅ Entry saved successfully!")

            st.markdown("**Saved values and where they came from:**")
            c1, c2, c3, c4 = st.columns(4)

            def show_metric(col, label, value, source):
                icon = (
                    "🖊️ Manual"   if source == "Manual"   else
                    "🤖 AI"       if source == "AI"        else
                    "⚠️ Fallback"
                )
                with col:
                    st.metric(label, value)
                    st.caption(icon)

            show_metric(c1, "🏷️ Title",  final_title,  src_title)
            show_metric(c2, "💰 Price",  final_price,  src_price)
            show_metric(c3, "🏪 Seller", final_seller, src_seller)
            show_metric(c4, "📦 Stock",  final_stock,  src_stock)

            st.info(f"🔑 **Tracking ID:** `{tracking_id}`")

            if "pasted_images" in st.session_state:
                st.session_state.pasted_images = []

            st.rerun()


    # =========================================================================
    # SECTION B — PRICE HISTORY DASHBOARD
    # =========================================================================
    st.divider()
    st.subheader("📊 Price History Dashboard")

    if price_history_df.empty:
        st.info("📭 No price history yet. Upload your first screenshot above!")
        return

    # Summary stats
    st.markdown("#### Overview")
    s1, s2, s3 = st.columns(3)
    with s1: st.metric("Total Records",    len(price_history_df))
    with s2: st.metric("Products Tracked", price_history_df["tracking_id"].nunique())
    with s3: st.metric("Last Updated",     price_history_df["timestamp"].iloc[-1])

    st.divider()

    # Search + dropdown
    st.markdown("#### 🔍 View History for a Specific Product")

    all_ids = price_history_df["tracking_id"].unique().tolist()

    search_query = st.text_input(
        label="🔎 Search by product name or seller",
        placeholder="e.g. Sonos  or  ASUS  or  Official Store",
        help="Type any word to filter the dropdown below. Leave blank to show all.",
        key="product_search",
    )

    if search_query.strip():
        filtered_ids = [
            tid for tid in all_ids
            if search_query.strip().lower() in tid.lower()
        ]
    else:
        filtered_ids = all_ids

    if search_query.strip():
        if filtered_ids:
            st.caption(f"✅ {len(filtered_ids)} match(es) found out of {len(all_ids)} products.")
        else:
            st.warning(f"⚠️ No products matched \"{search_query}\". Try a different keyword.")

    dropdown_options = filtered_ids if filtered_ids else all_ids

    selected_id = st.selectbox(
        label="Select a Product + Seller combination:",
        options=dropdown_options,
        help="Use the search box above to filter this list.",
    )

    filtered = price_history_df[
        price_history_df["tracking_id"] == selected_id
    ].copy()

    # Price chart
    st.markdown(f"##### Price Over Time — `{selected_id}`")
    try:
        filtered["price_numeric"] = pd.to_numeric(
            filtered["price"].str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        )
        if filtered["price_numeric"].notna().any():
            st.line_chart(
                filtered.set_index("timestamp")["price_numeric"],
                use_container_width=True,
            )
        else:
            st.warning("Could not draw chart — prices are not in a numeric format.")
    except Exception as e:
        st.warning(f"Chart error: {e}")

    # History table
    st.markdown("##### Entry History")
    show_cols = [
        "timestamp", "price", "quantity_left", "seller_name",
        "num_screenshots", "field_sources", "product_url",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    st.divider()

    # Export
    st.markdown("#### 📋 All Records & Export")

    if st.checkbox("Show full raw data table"):
        st.dataframe(price_history_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download Full History as CSV",
        data=price_history_df.to_csv(index=False).encode("utf-8"),
        file_name="shopee_price_history.csv",
        mime="text/csv",
        help="Download all your saved price records as a spreadsheet file.",
    )

    st.divider()
    st.caption(
        f"Shopee Price Tracker v5 • Model: {GEMINI_MODEL} • "
        "Storage: Local CSV • Powered by Streamlit"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
