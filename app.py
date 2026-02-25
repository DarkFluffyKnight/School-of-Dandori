import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI

import utils.rag as rag
from utils.getters import (
    get_all_categories,
    get_all_instructors,
    get_all_locations,
    get_all_unique_skills,
    load_and_clean_data,
)

# --- CONFIGURATION & API SETUP ---
# For Google Cloud deployment, you can set this in the Cloud Console.
# For local testing, ensure your key is available.

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    if "chat" not in st.session_state:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction="""You are the School of Dandori Assistant. 
Your goal is to help users find courses.
Maintain a whimsical, helpful tone.

---
INSTRUCTIONS FOR HANDLING DATA:
1. DATA CONTEXT contains the courses currently retrieved from the database.
2. **STRICT FILTERING:** If a user asks a follow-up question (e.g., "Which of those..."), you must ONLY list courses that satisfy BOTH the new criteria AND all previous criteria discussed in the chat.
3. **CONTEXTUAL CONTINUITY:** If the DATA CONTEXT contains new courses that do not match the previous topic (e.g., if they are in Devon but don't involve Wool), you MUST EXCLUDE them.
4. If no courses in the provided DATA CONTEXT meet the combined criteria, do not make them up. Instead, suggest they check the 'Discovery Gallery' and filters.

---
PROMPT FORMAT:
DATA CONTEXT:
[Documents]

USER QUESTION:
[The user question]""",
        )
        st.session_state.chat = model.start_chat()
else:
    st.error("Missing Gemini API Key! Please check your .env file.")


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("Missing OpenAI API Key! Please check your .env file.")

ENDPOINT = os.getenv("ENDPOINT", "https://openrouter.ai/api/v1")

chat_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=ENDPOINT)

# Page Setup
st.set_page_config(page_title="School of Dandori | Course Portal", layout="wide")

df = load_and_clean_data()

if "collection" not in st.session_state:
    st.session_state.collection = rag.get_collection(collection_name="pdf_data")

if "chat_client" not in st.session_state:
    st.session_state.chat_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=ENDPOINT)


# Visual Styling (Kept identical to your version)
st.markdown(
    """
    <style>
    .skill-tag {
        display: inline-block;
        background-color: #f0f4f8;
        color: #334e68;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px;
        font-size: 0.85rem;
        border: 1px solid #bcccdc;
    }
    .price-badge {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1b4332;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- RAG CHATBOT LOGIC (New) ---
def get_chatbot_response(user_query, df):
    """Retrieves relevant courses and generates a response via Gemini."""
    try:
        # 1. Retrieval Logic (Search)
        # We search course name and description for matches
        mask = df["course_name"].str.contains(user_query, case=False, na=False) | df[
            "course_description"
        ].str.contains(user_query, case=False, na=False)
        matches = df[mask].head(3)

        context = ""
        if not matches.empty:
            context = "Here are the most relevant courses from our catalog:\n"
            for _, row in matches.iterrows():
                context += f"- {row['course_name']} in {row['location']}. Cost: £{row['cost']}. Description: {row['course_description']}\n"

        # 2. Model Initialization
        # Use the full model path from available models
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""You are the School of Dandori Assistant. 
        Your goal is to help users find courses. 
        
        DATA CONTEXT:
        {context if context else 'No specific matches found in the catalog.'}
        
        USER QUESTION:
        {user_query}
        
        INSTRUCTIONS:
        - If courses are found in the DATA CONTEXT, describe them.
        - If no courses are found, suggest they check our 'Discovery Gallery' and filters.
        - Maintain a whimsical, helpful tone."""

        # 3. Generate content
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        # If it fails, we want to see the specific error
        return f"Dandori Error: {str(e)}"


def main():
    """Main app function"""
    # --- MAIN APP ---
    try:
        # df = load_and_clean_data()

        # chunks = rag.generate_chunks_from_dataframe(df=df)
        # collection = rag.create_collection(
        #     collection_name="pdf_data",
        #     api_key=OPENROUTER_API_KEY,
        #     base_url=ENDPOINT,
        # )
        # rag.add_chunks_to_collection(collection=collection, chunks=chunks)

        # --- SESSION STATE INITIALIZATION ---
        if "selected_skills" not in st.session_state:
            st.session_state.selected_skills = []
        if "selected_instructor" not in st.session_state:
            st.session_state.selected_instructor = []
        if "selected_category" not in st.session_state:
            st.session_state.selected_category = []
        if "selected_location" not in st.session_state:
            st.session_state.selected_location = []
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "favorites" not in st.session_state:
            st.session_state.favorites = []

        # --- Sidebar (Kept identical) ---
        st.sidebar.title("🌿 Dandori Menu")

        # Favorites section at the top
        with st.sidebar.expander(
            f"⭐ My Favorites ({len(st.session_state.favorites)})", expanded=False
        ):
            if st.session_state.favorites:
                for fav_id in st.session_state.favorites:
                    fav_course = df[df["class_id"] == fav_id]
                    if not fav_course.empty:
                        course_name = fav_course.iloc[0]["course_name"]
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"• {course_name}")
                        with col2:
                            if st.button(
                                "❌",
                                key=f"remove_fav_{fav_id}",
                                help="Remove from favorites",
                            ):
                                st.session_state.favorites.remove(fav_id)
                                st.rerun()
            else:
                st.write(
                    "No favorites yet. Click the ⭐ button on courses to add them!"
                )

        st.sidebar.divider()
        view_mode = st.sidebar.radio(
            "View Mode:", ["Discovery Gallery", "Data Table View", "My Favorites"]
        )
        st.sidebar.divider()

        st.sidebar.subheader("Filter Your Search")
        search_query = st.sidebar.text_input("Search keywords:", "")

        # Sort by dropdown
        sort_by = st.sidebar.selectbox(
            "Sort by:",
            options=[
                "Course Name (A-Z)",
                "Course Name (Z-A)",
                "Price (Low to High)",
                "Price (High to Low)",
                "Location (A-Z)",
                "Instructor (A-Z)",
            ],
        )

        slider_price = st.sidebar.slider(
            label="Price",
            min_value=float(df["cost"].min()),
            max_value=float(df["cost"].max()),
            value=(float(df["cost"].min()), float(df["cost"].max())),
        )

        if st.sidebar.button("Clear All Filters", use_container_width=True):
            st.session_state.selected_skills = []
            st.session_state.selected_instructor = []
            st.session_state.selected_category = []
            st.session_state.selected_location = []
            st.rerun()

        selected_location = st.sidebar.multiselect(
            "Location:",
            options=get_all_locations(df),
            default=st.session_state.selected_location,
        )
        selected_category = st.sidebar.multiselect(
            "Course Category:",
            options=get_all_categories(df),
            default=st.session_state.selected_category,
        )
        selected_instructor = st.sidebar.multiselect(
            "Course Instructor:",
            options=get_all_instructors(df),
            default=st.session_state.selected_instructor,
        )
        selected_skills = st.sidebar.multiselect(
            "Skills Developed:",
            options=get_all_unique_skills(df),
            default=st.session_state.selected_skills,
        )

        # State Sync
        if (
            selected_skills != st.session_state.selected_skills
            or selected_instructor != st.session_state.selected_instructor
            or selected_category != st.session_state.selected_category
            or selected_location != st.session_state.selected_location
        ):
            st.session_state.selected_skills = selected_skills
            st.session_state.selected_instructor = selected_instructor
            st.session_state.selected_category = selected_category
            st.session_state.selected_location = selected_location
            st.rerun()

        # --- Filter Logic (Kept identical) ---
        filtered_df = df.copy()
        if st.session_state.selected_location:
            filtered_df = filtered_df[
                filtered_df["location"].isin(st.session_state.selected_location)
            ]
        if st.session_state.selected_category:
            filtered_df = filtered_df[
                filtered_df["course_type"].isin(st.session_state.selected_category)
            ]
        if st.session_state.selected_instructor:
            filtered_df = filtered_df[
                filtered_df["instructor"].isin(st.session_state.selected_instructor)
            ]
        if st.session_state.selected_skills:
            filtered_df = filtered_df[
                filtered_df["skills_developed"].apply(
                    lambda s: any(sk in s for sk in st.session_state.selected_skills)
                )
            ]
        if search_query:
            filtered_df = filtered_df[
                filtered_df["course_name"].str.contains(search_query, case=False)
                | filtered_df["course_description"].str.contains(
                    search_query, case=False
                )
            ]
        filtered_df = filtered_df[
            (filtered_df["cost"] <= slider_price[1])
            & (filtered_df["cost"] >= slider_price[0])
        ]

        # Apply sorting
        if sort_by == "Course Name (A-Z)":
            filtered_df = filtered_df.sort_values("course_name", ascending=True)
        elif sort_by == "Course Name (Z-A)":
            filtered_df = filtered_df.sort_values("course_name", ascending=False)
        elif sort_by == "Price (Low to High)":
            filtered_df = filtered_df.sort_values("cost", ascending=True)
        elif sort_by == "Price (High to Low)":
            filtered_df = filtered_df.sort_values("cost", ascending=False)
        elif sort_by == "Location (A-Z)":
            filtered_df = filtered_df.sort_values("location", ascending=True)
        elif sort_by == "Instructor (A-Z)":
            filtered_df = filtered_df.sort_values("instructor", ascending=True)

        # --- TABS INTERFACE ---
        # We use Tabs to keep the old UI perfectly preserved on the first tab.
        tab_portal, tab_ai = st.tabs(["🏛️ Course Portal", "🤖 Dandori Assistant"])

        with tab_portal:
            if view_mode == "My Favorites":
                st.title("⭐ My Favorite Courses")

                if st.session_state.favorites:
                    favorites_df = df[df["class_id"].isin(st.session_state.favorites)]
                    st.write(f"You have **{len(favorites_df)}** favorite courses.")
                    st.divider()

                    for r_num, row in favorites_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.subheader(row["course_name"])
                                st.write(
                                    f"📍 **{row['location']}** | 👤 **{row['instructor']}**"
                                )
                                desc = row["course_description"]
                                if len(desc) > 150:
                                    st.write(f"{desc[:150]}...")
                                    with st.expander("Read full description"):
                                        st.write(desc)
                                else:
                                    st.write(desc)

                                st.write("**Skills:**")
                                skill_cols = st.columns(
                                    min(len(row["skills_developed"]), 5)
                                )
                                for idx, skill in enumerate(row["skills_developed"]):
                                    with skill_cols[idx % 5]:
                                        is_selected = (
                                            skill in st.session_state.selected_skills
                                        )
                                        if st.button(
                                            skill,
                                            key=f"fav_sk_{r_num}_{idx}",
                                            type=(
                                                "primary"
                                                if is_selected
                                                else "secondary"
                                            ),
                                            use_container_width=True,
                                        ):
                                            if is_selected:
                                                st.session_state.selected_skills.remove(
                                                    skill
                                                )
                                            else:
                                                st.session_state.selected_skills.append(
                                                    skill
                                                )
                                            st.rerun()

                            with col2:
                                st.markdown(
                                    f'<p class="price-badge">£{row["cost"]:.2f}</p>',
                                    unsafe_allow_html=True,
                                )
                                with st.expander("Learning Objectives"):
                                    for obj in row["learning_objectives"]:
                                        st.write(f"• {obj}")

                                col_book, col_unfav = st.columns(2)
                                with col_book:
                                    st.button("Book Now", key=f"fav_btn_{r_num}")
                                with col_unfav:
                                    if st.button(
                                        "💔",
                                        key=f"unfav_{r_num}",
                                        help="Remove from favorites",
                                    ):
                                        st.session_state.favorites.remove(
                                            row["class_id"]
                                        )
                                        st.rerun()
                            st.divider()
                else:
                    st.info(
                        "You haven't added any favorites yet. Browse the Discovery Gallery and click the ⭐ button to save courses!"
                    )

            elif view_mode == "Discovery Gallery":
                st.title("School of Dandori")
                st.write(f"Showing **{len(filtered_df)}** whimsical classes.")
                st.divider()

                items_per_page = 20

                # Calculate total pages
                total_pages = (len(filtered_df) // items_per_page) + (
                    1 if len(filtered_df) % items_per_page > 0 else 0
                )

                # Add a page selector at the top (and bottom)
                if total_pages > 1:
                    current_page = st.number_input(
                        f"Page (1 of {total_pages})",
                        min_value=1,
                        max_value=total_pages,
                        step=1,
                    )
                else:
                    current_page = 1

                # Calculate start and end indices for the current page
                start_idx = (current_page - 1) * items_per_page
                end_idx = start_idx + items_per_page

                # Slice the dataframe for the current page
                page_df = filtered_df.iloc[start_idx:end_idx]

                for r_num, row in page_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader(row["course_name"])
                            st.write(
                                f"📍 **{row['location']}** | 👤 **{row['instructor']}**"
                            )
                            desc = row["course_description"]
                            if len(desc) > 150:
                                st.write(f"{desc[:150]}...")
                                with st.expander("Read full description"):
                                    st.write(desc)
                            else:
                                st.write(desc)

                            st.write("**Skills:**")
                            skill_cols = st.columns(
                                min(len(row["skills_developed"]), 5)
                            )
                            for idx, skill in enumerate(row["skills_developed"]):
                                with skill_cols[idx % 5]:
                                    is_selected = (
                                        skill in st.session_state.selected_skills
                                    )
                                    if st.button(
                                        skill,
                                        key=f"sk_{r_num}_{idx}",
                                        type="primary" if is_selected else "secondary",
                                        use_container_width=True,
                                    ):
                                        if is_selected:
                                            st.session_state.selected_skills.remove(
                                                skill
                                            )
                                        else:
                                            st.session_state.selected_skills.append(
                                                skill
                                            )
                                        st.rerun()

                        with col2:
                            st.markdown(
                                f'<p class="price-badge">£{row["cost"]:.2f}</p>',
                                unsafe_allow_html=True,
                            )
                            with st.expander("Learning Objectives"):
                                for obj in row["learning_objectives"]:
                                    st.write(f"• {obj}")

                            # Favorite button and Book button
                            col_fav, col_book = st.columns([1, 2])
                            with col_fav:
                                is_favorite = (
                                    row["class_id"] in st.session_state.favorites
                                )
                                if st.button(
                                    "⭐" if is_favorite else "☆",
                                    key=f"fav_toggle_{r_num}",
                                    help=(
                                        "Add to favorites"
                                        if not is_favorite
                                        else "Remove from favorites"
                                    ),
                                    use_container_width=True,
                                ):
                                    if is_favorite:
                                        st.session_state.favorites.remove(
                                            row["class_id"]
                                        )
                                    else:
                                        st.session_state.favorites.append(
                                            row["class_id"]
                                        )
                                    st.rerun()
                            with col_book:
                                st.button(
                                    "Book Now",
                                    key=f"btn_{r_num}",
                                    use_container_width=True,
                                )
                        st.divider()
            else:
                st.title("Admin Data View")
                display_df = filtered_df[
                    [
                        "course_name",
                        "instructor",
                        "location",
                        "course_type",
                        "cost",
                        "skills_developed",
                        "course_description",
                    ]
                ]
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "class_id": "ID",
                        "course_name": "Course Name",
                        "instructor": "Instructor",
                        "course_type": "Category",
                        "location": "Location",
                        "cost": st.column_config.NumberColumn("Cost", format="£ %.2f"),
                        "skills_developed": "Skills",
                        "course_description": "Full Description",
                    },
                )
                st.download_button(
                    "📥 Download Filtered CSV",
                    data=filtered_df.to_csv(index=False),
                    file_name="dandori_export.csv",
                    mime="text/csv",
                )

        with tab_ai:
            st.title("🤖 Chat with Arthur's Assistant")
            st.write("Ask me anything about our curriculum!")

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat Input
            if prompt := st.chat_input("I'm looking for a baking class in York..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):

                        # response = get_chatbot_response(prompt, df)
                        # response = rag.query_llm_with_rag(
                        #     chat_client=st.session_state.chat_client,
                        #     collection_name="pdf_data",
                        #     collection=st.session_state.collection,
                        #     query=prompt,
                        #     history=st.session_state.messages,
                        # )

                        response = rag.query_gemini_with_rag(
                            chat=st.session_state.chat,
                            collection_name="pdf_data",
                            collection=st.session_state.collection,
                            query=prompt,
                        )

                        st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    except FileNotFoundError:
        st.error("Missing Data: Please ensure course_data.csv is present.")


if __name__ == "__main__":
    main()
