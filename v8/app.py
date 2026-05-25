# =============================================================================
# SHOPEE PRICE TRACKER - app.py  (v5 — uses gemini-2.5-flash-lite, free tier)
# =============================================================================
# MODEL USED: gemini-2.5-flash-lite
#   • Free tier: 1,000 requests/day, no credit card needed
#   • Supports images (multimodal) — perfect for reading screenshots
#   • Stable, not deprecated — safe to use in 2026 and beyond
#   • Uses the new 'google-genai' SDK (stable v1 API, not the old v1beta)
#
# FEATURES:
#   • Upload as many screenshots as needed (AI reads all together)
#   • Manual override fields for Title, Price, Seller if AI misses them
#   • Stock/Quantity is fully optional — saves as N/A if not found
#   • Price history saved to a local CSV file (never overwrites old data)
#   • Dashboard with price-over-time chart and full history table
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

# New official Google AI SDK — replaces the old 'google-generativeai' package.
# This uses the stable v1 API and supports all current Gemini models.
from google import genai
from google.genai import types


# =============================================================================
# CONFIGURATION
# =============================================================================

# The CSV file where all price history is stored.
# Created automatically when you save your first entry.
CSV_FILE = "price_history.csv"

# Column headers for the price history spreadsheet.
CSV_COLUMNS = [
    "timestamp",        # Date + time the entry was saved
    "tracking_id",      # "Product Title | Seller Name" — unique per product+seller
    "product_title",    # The product's full name
    "seller_name",      # The shop/seller name
    "price",            # Current price (e.g. "RM 25.90")
    "quantity_left",    # Stock remaining — "N/A" if not found
    "product_url",      # Shopee URL (for your reference only)
    "num_screenshots",  # How many screenshots were uploaded
    "field_sources",    # Which fields came from AI vs manual input
]

# The Gemini model to use.
# gemini-2.5-flash-lite = free tier, 1000 req/day, supports images.
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Values the AI returns when it cannot find something.
# We check against this list to know if a field needs manual input.
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
    Creates a Gemini API client using the secret key stored in Streamlit Secrets.
    With the new google-genai SDK, we create a Client object and reuse it
    throughout the app — instead of calling a global configure() function.

    Returns the client object on success, or None on failure.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client  = genai.Client(api_key=api_key)
        return client
    except KeyError:
        st.error(
            "🔑 **API Key Not Found!**\n\n"
            "Please add your Gemini API key to Streamlit Secrets.\n\n"
            "**Running locally?** Edit `.streamlit/secrets.toml` and add:\n"
            "```\nGEMINI_API_KEY = \"your-key-here\"\n```\n\n"
            "**On Streamlit Cloud?** Go to App Settings → Secrets and add the same line."
        )
        return None
    except Exception as e:
        st.error(f"❌ Error connecting to Gemini: {e}")
        return None


def load_price_history():
    """
    Reads the CSV file and returns it as a pandas DataFrame (a table in memory).
    If the file doesn't exist yet, returns an empty table with the correct columns.
    Also adds any new columns from newer app versions so old data loads cleanly.
    """
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Patch in any columns added in newer versions that old CSV won't have
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = "N/A"
        return df
    return pd.DataFrame(columns=CSV_COLUMNS)


def save_new_entry(record: dict):
    """
    Appends one new row to the bottom of the CSV file.
    NEVER deletes or overwrites existing rows — history is always preserved.

    record = a Python dictionary whose keys match CSV_COLUMNS exactly.
    """
    new_df     = pd.DataFrame([record])
    file_exists = os.path.exists(CSV_FILE)
    # mode='a'           → append to end, don't erase existing content
    # header=not file_exists → write column names only if this is a new file
    # index=False        → don't write pandas row numbers into the file
    new_df.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)


def build_image_part(image_bytes: bytes):
    """
    Converts raw image bytes into a types.Part object that the google-genai
    SDK can send to the Gemini API. Also detects the correct MIME type
    (e.g. "image/jpeg" for JPEGs, "image/png" for PNGs).
    """
    pil_img   = Image.open(io.BytesIO(image_bytes))
    fmt       = pil_img.format if pil_img.format else "JPEG"
    mime_map  = {
        "JPEG": "image/jpeg",
        "PNG":  "image/png",
        "WEBP": "image/webp",
        "GIF":  "image/gif",
    }
    mime_type = mime_map.get(fmt, "image/jpeg")
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def extract_data_with_gemini(client, all_image_bytes: list) -> tuple:
    """
    Sends ALL uploaded screenshots to Gemini AI in one single request.
    The AI is told how many images it's receiving and reads all of them
    together — so information spread across multiple screenshots is combined.

    Parameters:
        client           — the Gemini Client created by get_gemini_client()
        all_image_bytes  — list of raw bytes, one entry per uploaded screenshot

    Returns:
        (result_dict, None)          on success
        (None, "error description")  on failure
    """
    num_images = len(all_image_bytes)

    # Describe to the AI how many images it's looking at
    if num_images == 1:
        image_context = "You are given 1 screenshot of a Shopee product page."
    else:
        image_context = (
            f"You are given {num_images} screenshots of the SAME Shopee product page, "
            f"taken from different scroll positions (e.g. top, middle, bottom). "
            f"Treat all of them together as one complete view of the page."
        )

    # The prompt — very specific instructions telling the AI exactly what to
    # extract and exactly what format to return it in (raw JSON, nothing else).
    prompt = f"""
You are a data extraction assistant. {image_context}

Using ALL the images together, extract the following four pieces of information:

1. product_title  — The full product name/title shown on the page.
2. price          — The current selling price including its currency symbol.
                    Examples: "RM 25.90", "PHP 199", "$12.50".
                    Use the discounted/sale price if two prices are shown.
                    If a price range is shown (e.g. RM10–RM20), return the lower value.
3. seller_name    — The name of the shop or seller.
                    Usually found near a "Chat Now", "Follow", or "Visit Shop" button.
4. quantity_left  — Remaining stock level.
                    Look for phrases like "X pieces available", "Stock: X", etc.
                    Return "N/A" if stock information is not visible anywhere.

Return ONLY a valid JSON object with exactly these four keys.
Do NOT include any explanation, markdown formatting, or code fences.
Just the raw JSON and nothing else:
{{
  "product_title": "...",
  "price": "...",
  "seller_name": "...",
  "quantity_left": "..."
}}
"""

    try:
        # Build the list of content parts:
        # all image parts first, then the text prompt at the end.
        contents = [build_image_part(b) for b in all_image_bytes] + [prompt]

        # Send everything to Gemini in one API call.
        # client.models.generate_content() is the new SDK's way of calling the AI.
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )

        raw_text = response.text.strip()

        # Extract the JSON block from the response.
        # re.DOTALL allows '.' to match newlines, so multi-line JSON is captured.
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            return None, f"AI did not return valid JSON.\nRaw response:\n{raw_text}"

        data   = json.loads(match.group())
        result = {
            "product_title": data.get("product_title", "Not Found"),
            "price":         data.get("price",         "Not Found"),
            "seller_name":   data.get("seller_name",   "Not Found"),
            "quantity_left": data.get("quantity_left", "N/A"),
        }
        return result, None

    except json.JSONDecodeError as e:
        return None, f"Could not parse AI response as JSON: {e}"
    except Exception as e:
        return None, f"Gemini API error: {e}"


def is_missing(value: str) -> bool:
    """
    Returns True if a value is blank or a known 'not found' placeholder.
    Used to decide which fields still need manual input after AI extraction.
    """
    if not value:
        return True
    return value.strip().lower() in MISSING_VALUES


def create_tracking_id(title: str, seller: str) -> str:
    """
    Builds the unique tracking key for a product+seller combination.
    Example: "Wireless Earbuds Pro | TechShopMY Official Store"
    Two different sellers selling the same product get separate tracking lines.
    """
    return f"{title.strip()} | {seller.strip()}"


# =============================================================================
# MAIN APP — builds everything the user sees on the page
# =============================================================================

def main():

    # --- Page header ---
    st.title("🛒 Shopee Price Tracker")
    st.markdown(
        "Upload **one or more** screenshots of a Shopee product page. "
        "The AI reads all of them together and extracts the price data. "
        "Any field the AI misses can be filled in manually before saving."
    )
    st.divider()

    # --- Connect to Gemini (stop the app if connection fails) ---
    client = get_gemini_client()
    if client is None:
        st.stop()

    # --- Load existing price history ---
    price_history_df = load_price_history()


    # =========================================================================
    # SECTION A — UPLOAD & EXTRACT
    # =========================================================================
    with st.expander("📸 **Track a New Price Entry**", expanded=True):

        # ------------------------------------------------------------------
        # Step 1 — Product URL (optional, for reference only)
        # ------------------------------------------------------------------
        st.markdown("#### Step 1 — Product URL *(optional)*")
        product_url = st.text_input(
            label="Paste the Shopee product URL here (saved for your records, not used to fetch data)",
            placeholder="https://shopee.com.my/product/...",
        )

        st.divider()

        # ------------------------------------------------------------------
        # Step 2 — Add screenshots via file upload AND/OR clipboard paste
        # ------------------------------------------------------------------
        st.markdown("#### Step 2 — Add Screenshot(s)")
        st.markdown(
            "Add screenshots using **either or both** methods below. "
            "All added images are sent to the AI together."
        )

        # ── Method A: Clipboard Paste (Mac-friendly) ──────────────────────
        # On Mac: press Cmd+Ctrl+Shift+4 to copy a region to clipboard,
        # then click the Paste button below to load it into the app.
        # The streamlit-paste-button library handles all clipboard reading.
        st.markdown(
            "**📋 Method A — Paste from Clipboard** *(Mac: `Cmd+Ctrl+Shift+4` → click button)*"
        )

        # Import the paste button from the streamlit-paste-button library.
        # We import it here (inside the function) so the app still loads
        # even if the library isn't installed yet — it will show a clear message.
        try:
            from streamlit_paste_button import paste_image_button

            # This renders a clickable paste button on the page.
            # When clicked, it reads whatever image is in the clipboard.
            # 'paste_result' holds the pasted image data (or None if nothing was pasted).
            paste_result = paste_image_button(
                label="📋 Click here to paste screenshot from clipboard",
                background_color="#2d6a4f",   # Dark green button colour
                hover_background_color="#1b4332",
                key="paste_btn",
            )

            # paste_result.image_data is a PIL Image object when something was pasted.
            # We store it in session_state so it persists across reruns.
            # (Streamlit reruns the whole script on every interaction, so we need
            #  session_state to "remember" things between those reruns.)
            if paste_result and paste_result.image_data is not None:
                # Convert the pasted PIL Image to PNG bytes so we can store and display it.
                paste_buffer = io.BytesIO()
                paste_result.image_data.save(paste_buffer, format="PNG")
                pasted_bytes = paste_buffer.getvalue()

                # Use session_state as a list to accumulate multiple pasted images.
                # session_state is Streamlit's way of storing values that survive reruns.
                if "pasted_images" not in st.session_state:
                    st.session_state.pasted_images = []

                # Only add it if it isn't already in the list (avoid duplicates on rerun).
                if pasted_bytes not in st.session_state.pasted_images:
                    st.session_state.pasted_images.append(pasted_bytes)
                    st.success(
                        f"✅ Screenshot pasted! "
                        f"({len(st.session_state.pasted_images)} pasted image(s) ready)"
                    )

            # Show all currently pasted images as previews.
            if "pasted_images" in st.session_state and st.session_state.pasted_images:
                st.markdown(
                    f"*{len(st.session_state.pasted_images)} pasted image(s):*"
                )
                paste_cols = st.columns(min(len(st.session_state.pasted_images), 4))
                for i, pb in enumerate(st.session_state.pasted_images):
                    with paste_cols[i % 4]:
                        st.image(pb, caption=f"Pasted {i + 1}", use_container_width=True)

                # A small button to clear all pasted images if the user wants to start over.
                if st.button("🗑️ Clear pasted images", key="clear_paste"):
                    st.session_state.pasted_images = []
                    st.rerun()

        except ImportError:
            # If the library isn't installed, show a clear message instead of crashing.
            st.warning(
                "⚠️ **Clipboard paste not available.** "
                "Install it by running this in your terminal:\n"
                "```\npip install streamlit-paste-button\n```\n"
                "Then restart the app. You can still upload files using Method B below."
            )
            paste_result = None

        st.markdown("---")

        # ── Method B: File Upload ─────────────────────────────────────────
        # The classic file picker — works on all devices including mobile.
        st.markdown(
            "**📁 Method B — Upload File(s)** *(works on all devices including mobile)*"
        )

        # accept_multiple_files=True lets the user pick more than one image at once.
        # On mobile, they can select multiple photos from their gallery.
        uploaded_files = st.file_uploader(
            label="Upload screenshot(s) — JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help=(
                "💡 Tip: On mobile, use your phone's scrolling/long screenshot feature "
                "to capture the full page in one image if possible."
            ),
        )

        # Show thumbnail previews of uploaded files
        if uploaded_files:
            st.markdown(f"*{len(uploaded_files)} uploaded file(s):*")
            cols_per_row = min(len(uploaded_files), 4)
            preview_cols = st.columns(cols_per_row)
            for i, uf in enumerate(uploaded_files):
                with preview_cols[i % cols_per_row]:
                    st.image(uf, caption=f"Upload {i + 1}", use_container_width=True)

        st.divider()

        # ------------------------------------------------------------------
        # Step 3 — Manual override fields
        # Always visible so you can fill them in before OR after clicking Analyze.
        # Your typed values ALWAYS take priority over the AI's extracted values.
        # ------------------------------------------------------------------
        st.markdown("#### Step 3 — Manual Overrides *(fill in if AI misses anything)*")
        st.markdown(
            "Leave these blank to let the AI fill them automatically. "
            "If the AI gets something wrong or can't find it, type the correct "
            "value here. **Your typed value always overrides the AI.**\n\n"
            "Stock is optional — if left blank it saves as N/A and never blocks saving."
        )

        oc1, oc2 = st.columns(2)
        with oc1:
            manual_title = st.text_input(
                label="📦 Product Title",
                placeholder="e.g. Wireless Bluetooth Earbuds Pro Max",
                key="manual_title",
                help="Required field. Type the full product name if AI got it wrong.",
            )
            manual_price = st.text_input(
                label="💰 Price",
                placeholder="e.g. RM 25.90",
                key="manual_price",
                help="Required field. Include the currency symbol (RM, $, PHP, etc.).",
            )
        with oc2:
            manual_seller = st.text_input(
                label="🏪 Seller / Shop Name",
                placeholder="e.g. TechShopMY Official Store",
                key="manual_seller",
                help="Required field. Shop name as it appears on Shopee.",
            )
            manual_stock = st.text_input(
                label="📦 Stock / Quantity Left *(optional)*",
                placeholder="e.g. 47  — or leave blank for N/A",
                key="manual_stock",
                help="Optional. Leave blank if not visible on the page.",
            )

        st.divider()

        # ------------------------------------------------------------------
        # Step 4 — Analyze button
        # Enabled when at least one image exists from EITHER source.
        # ------------------------------------------------------------------
        st.markdown("#### Step 4 — Analyze & Save")

        # Collect images from BOTH sources into one combined list.
        # Pasted images come from session_state; uploaded files are read here.
        pasted_bytes_list  = st.session_state.get("pasted_images", [])
        uploaded_bytes_list = [uf.read() for uf in uploaded_files] if uploaded_files else []

        # Combine both lists — AI will receive all images together.
        all_image_bytes_preview = pasted_bytes_list + uploaded_bytes_list
        total_images = len(all_image_bytes_preview)

        no_images      = total_images == 0
        analyze_clicked = st.button(
            label=f"🤖 Analyze {total_images} Screenshot(s) with AI & Save" if total_images > 0
                  else "🤖 Analyze Screenshot(s) with AI & Save",
            type="primary",
            use_container_width=True,
            disabled=no_images,
        )
        if no_images:
            st.caption("⬆️ Paste or upload at least one screenshot above to enable this button.")

        # ------------------------------------------------------------------
        # Processing — only runs when the button is clicked
        # ------------------------------------------------------------------
        if analyze_clicked and not no_images:

            # Use the combined list built above (pasted + uploaded).
            # We already read uploaded_files into uploaded_bytes_list above,
            # so we just combine the two lists again here.
            all_image_bytes = pasted_bytes_list + uploaded_bytes_list

            with st.spinner(
                f"Sending {len(all_image_bytes)} screenshot(s) to Gemini AI "
                f"({GEMINI_MODEL})… Please wait 5–25 seconds…"
            ):
                ai_result, error_msg = extract_data_with_gemini(client, all_image_bytes)

            # --- Handle AI failure gracefully ---
            # Even if the AI fails entirely, the user can still save manually.
            if error_msg:
                st.error(f"❌ AI extraction failed:\n\n{error_msg}")
                st.info(
                    "💡 **You can still save manually.** "
                    "Fill in all fields in Step 3 above and click Analyze again. "
                    "The app will use your typed values and skip the AI."
                )
                # Set all AI results to "Not Found" so the manual merge logic below still works
                ai_result = {
                    "product_title": "Not Found",
                    "price":         "Not Found",
                    "seller_name":   "Not Found",
                    "quantity_left": "N/A",
                }

            # --- Merge AI results with manual overrides ---
            # Priority order: Manual input → AI result → fallback value
            def resolve(manual_val, ai_val, fallback="Not Found"):
                """
                Picks the best value for a field.
                Returns (final_value, source_label) where source_label is
                "Manual", "AI", or "fallback".
                """
                mv = manual_val.strip() if manual_val else ""
                if mv:                          # User typed something → use it
                    return mv, "Manual"
                elif not is_missing(ai_val):    # AI found something → use it
                    return ai_val, "AI"
                else:                           # Neither worked → use fallback
                    return fallback, "fallback"

            final_title,  src_title  = resolve(manual_title,  ai_result["product_title"])
            final_price,  src_price  = resolve(manual_price,  ai_result["price"])
            final_seller, src_seller = resolve(manual_seller, ai_result["seller_name"])
            # Stock: fallback is "N/A" — never blocks saving
            final_stock,  src_stock  = resolve(manual_stock,  ai_result["quantity_left"], "N/A")

            # --- Check that the three REQUIRED fields are present ---
            missing_required = []
            if is_missing(final_title):  missing_required.append("Product Title")
            if is_missing(final_price):  missing_required.append("Price")
            if is_missing(final_seller): missing_required.append("Seller Name")

            if missing_required:
                st.warning(
                    "⚠️ **The following required field(s) could not be determined:**\n\n"
                    + "\n".join(f"- **{f}**" for f in missing_required)
                    + "\n\nPlease fill them in manually in **Step 3** above, "
                    "then click **Analyze** again."
                )
                st.stop()  # Don't save an incomplete record

            # --- Build field source summary (stored in the history table) ---
            source_summary = (
                f"title={src_title}, price={src_price}, "
                f"seller={src_seller}, stock={src_stock}"
            )

            # --- Save the record to the CSV file ---
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

            # --- Show success and a summary of what was saved ---
            st.success("✅ Entry saved successfully!")

            st.markdown("**Saved values and where they came from:**")
            c1, c2, c3, c4 = st.columns(4)

            def show_metric(col, label, value, source):
                """Displays a metric card with a coloured source badge below."""
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

            # Clear the pasted images from memory now that they've been saved,
            # so the paste area is clean and ready for the next product.
            if "pasted_images" in st.session_state:
                st.session_state.pasted_images = []

            # Reload the page so the dashboard below shows the latest data
            st.rerun()


    # =========================================================================
    # SECTION B — PRICE HISTORY DASHBOARD
    # =========================================================================
    st.divider()
    st.subheader("📊 Price History Dashboard")

    if price_history_df.empty:
        st.info(
            "📭 No price history yet. "
            "Upload your first screenshot above to get started!"
        )
        return  # Nothing more to show — exit early

    # --- Summary stats ---
    st.markdown("#### Overview")
    s1, s2, s3 = st.columns(3)
    with s1: st.metric("Total Records",    len(price_history_df))
    with s2: st.metric("Products Tracked", price_history_df["tracking_id"].nunique())
    with s3: st.metric("Last Updated",     price_history_df["timestamp"].iloc[-1])

    st.divider()

    # --- Per-product history viewer ---
    st.markdown("#### 🔍 View History for a Specific Product")

    all_ids = price_history_df["tracking_id"].unique().tolist()

    # ── Search box ────────────────────────────────────────────────────────
    # The user types any part of the product name or seller name here.
    # The dropdown below is instantly filtered to only show matching results.
    # Leaving the search box blank shows ALL products in the dropdown.
    search_query = st.text_input(
        label="🔎 Search by product name or seller",
        placeholder="e.g. Sonos  or  Just IT  or  monitor",
        help=(
            "Type any word from the product title or seller name to narrow down "
            "the dropdown list below. Case-insensitive. Leave blank to show all."
        ),
        key="product_search",
    )

    # Filter the list of tracking IDs based on the search query.
    # .lower() makes the search case-insensitive so 'sonos' matches 'Sonos'.
    if search_query.strip():
        filtered_ids = [
            tid for tid in all_ids
            if search_query.strip().lower() in tid.lower()
        ]
    else:
        filtered_ids = all_ids  # No search text -> show everything

    # Show how many results matched
    if search_query.strip():
        if filtered_ids:
            st.caption(f"✅ {len(filtered_ids)} match(es) found out of {len(all_ids)} products.")
        else:
            st.warning(
                f"⚠️ No products matched **\"{search_query}\"**. "
                "Try a shorter or different keyword."
            )

    # ── Dropdown (always visible, filtered by search above) ───────────────
    # If the search returned no results, fall back to showing all products
    # so the user is never left with a broken/empty dropdown.
    dropdown_options = filtered_ids if filtered_ids else all_ids

    selected_id = st.selectbox(
        label="Select a Product + Seller combination:",
        options=dropdown_options,
        help=(
            "Each entry is a unique Product Title + Seller Name pair. "
            "Use the search box above to filter this list."
        ),
    )

    filtered = price_history_df[
        price_history_df["tracking_id"] == selected_id
    ].copy()

    # Price chart — strips currency symbols to get a plain number for the Y axis
    st.markdown(f"##### Price Over Time — `{selected_id}`")
    try:
        filtered["price_numeric"] = pd.to_numeric(
            filtered["price"].str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",  # Turn unconvertible values into NaN instead of crashing
        )
        if filtered["price_numeric"].notna().any():
            st.line_chart(
                filtered.set_index("timestamp")["price_numeric"],
                use_container_width=True,
            )
        else:
            st.warning(
                "Could not draw a price chart — "
                "the prices in this history are not in a numeric format."
            )
    except Exception as e:
        st.warning(f"Chart could not be generated: {e}")

    # History table
    st.markdown("##### Entry History")
    show_cols = [
        "timestamp", "price", "quantity_left", "seller_name",
        "num_screenshots", "field_sources", "product_url",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    st.divider()

    # --- Export all data ---
    st.markdown("#### 📋 All Records & Export")

    if st.checkbox("Show full raw data table"):
        st.dataframe(price_history_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download Full History as CSV",
        data=price_history_df.to_csv(index=False).encode("utf-8"),
        file_name="shopee_price_history.csv",
        mime="text/csv",
        help="Downloads all your saved price records as a spreadsheet file.",
    )

    # --- Footer ---
    st.divider()
    st.caption(
        f"Shopee Price Tracker v5 • Model: {GEMINI_MODEL} (free tier) • "
        "Powered by Google Gemini AI & Streamlit • "
        "Data stored locally in price_history.csv"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
