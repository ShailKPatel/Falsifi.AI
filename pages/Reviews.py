import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import json
import os
from collections import Counter, deque
import re
import numpy as np

st.set_page_config(page_title="Falsifi.AI", layout="wide")


# -------------------------------
# Inject Global Dark Theme + Custom Text Selection
# -------------------------------
st.markdown("""
<style>
/* Global dark background and white text */
body {
    background-color: black !important;
    color: white !important;
}

/* All text elements white (except h1 which is overridden by class) */
h2, h3, h4, h5, h6, p, li, a, label {
    color: white;
}

/* Custom H1 color override */
.custom-h1 {
    color: #374151 !important;     /* Dark gray */
    font-weight: 800 !important;   /* Bold */
    font-size: 3.5rem !important;  /* Larger font size */
}


/* Custom text selection style */
::selection {
    background: white;
    color: black;
}
::-moz-selection {
    background: white;
    color: black;
}

/* Clean block spacing */
.block-container {
    padding: 2rem 1rem;
}

/* Module boxes */
.module-box {
    background-color: #111;
    padding: 1.5rem;
    border: 1px solid #444;
    border-radius: 10px;
    margin-bottom: 1rem;
    height: 100%;
}
.module-box a {
    color: white !important;
    font-weight: bold;
    text-decoration: none;
}

.custom-h1 {
    color: #4B5563 !important;     
    font-weight: 800 !important;   
    font-size: 3.75rem !important;  
    margin-bottom: 0rem !important;
    margin-top: 0rem !important;
}

.custom-subtitle {
    font-size: 1.25rem !important;
    color: white !important;
    margin-top: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section 
# -------------------------------
st.markdown("""
<h1 class="custom-h1">Reviews</h1>
""", unsafe_allow_html=True)


# File Paths
REVIEW_PATH = "reviews/recent_reviews.json"
WORD_COUNT_PATH = "reviews/word_count.json"
REVIEW_LIMIT = 6

# Expanded filter words to prevent "lol" variations and vandalism
FILTER_WORDS = {
    "lol", "lolis", "laughing",   # Cover "laughing out loud" and variations
    "haha", "hehe", "lmao", "rofl", '3'    # Other informal laughter terms
}

# Load reviews
def load_reviews():
    if os.path.exists(REVIEW_PATH):
        with open(REVIEW_PATH, "r", encoding="utf-8") as file:
            return deque(json.load(file), maxlen=REVIEW_LIMIT)
    return deque(maxlen=REVIEW_LIMIT)

# Save reviews
def save_reviews(reviews):
    with open(REVIEW_PATH, "w", encoding="utf-8") as file:
        json.dump(list(reviews), file)

# Load word count
def load_word_count():
    if os.path.exists(WORD_COUNT_PATH):
        with open(WORD_COUNT_PATH, "r", encoding="utf-8") as file:
            return Counter(json.load(file))
    return Counter()

# Save word count
def save_word_count(counter):
    with open(WORD_COUNT_PATH, "w", encoding="utf-8") as file:
        json.dump(dict(counter), file)

# Filter words: stop words, "lol" variations, single letters, and digits
def should_count_word(word):
    # Use regex to catch "lol" with special characters (e.g., "lol!")
    if re.search(r'l+o+l+[!@#$%^&*]*', word.lower()):
        return False
    # Exclude single letters and digits
    if len(word) == 1 or word.isdigit():
        return False
    # Check against stop words and filter words
    return (word.lower() not in STOPWORDS and 
            word.lower() not in FILTER_WORDS and 
            not any(fw in word.lower() for fw in FILTER_WORDS))

# Check if review contains "lol" or filtered words
def contains_filtered_words(review):
    words = review.split()
    for word in words:
        # Check for "lol" variations with regex
        if re.search(r'l+o+l+[!@#$%^&*]*', word.lower()):
            return True
        # Check for other filter words
        if any(fw in word.lower() for fw in FILTER_WORDS) or word.lower() in FILTER_WORDS:
            return True 
    return False

# Initialize data
review_queue = load_reviews()
word_count = load_word_count()

with st.container(border=True):
    st.subheader("📜 Recent Reviews")
    if review_queue:
        for review in review_queue:
            st.write(f"✍️ {review}")
    else:
        st.write("No reviews yet. Be the first to leave your mark! ✨")

with st.container(border=True):
    st.markdown('<h2 style="margin-bottom: 0.5rem;">📊 Word Cloud</h2>', unsafe_allow_html=True)

    if word_count:
        # Filter word_count to remove stop words and filtered words
        filtered_word_count = Counter()
        for word, freq in word_count.items():
            if should_count_word(word):
                filtered_word_count[word.lower()] += freq

        if filtered_word_count:
            def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                colors = ["rgb(73,11,61)", "rgb(189,30,81)", "rgb(241,184,20)", "rgb(128,173,204)"]
                return np.random.choice(colors)

            # High-resolution word cloud generation
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="#FCFAEE",
                max_words=50,
                min_font_size=12,
                scale=8,
                stopwords=STOPWORDS.union(FILTER_WORDS),
                color_func=random_color_func
            ).generate_from_frequencies(filtered_word_count)

            # Center-aligned half-width display
            left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
            with center_col:
                fig, ax = plt.subplots(figsize=(5, 2.5))  # Half-screen display size
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        else:
            st.write("No valid words to display in the word cloud after filtering.")
    else:
        st.write("No words to display in the word cloud yet. Submit a review!")



    
# Submit new review
user_review = st.text_area("Got a complaint or suggestion? Drop it here", "")
if st.button("Submit Review"):
    if user_review:
        # Check if the review contains filtered words
        if not contains_filtered_words(user_review):
            # Only save and process the review if it doesn't contain filtered words
            review_queue.append(user_review)
            save_reviews(review_queue)

            # Process words for word count: remove punctuation, normalize case
            # Removes 's, 'd, and other apostrophes from words
            processed_review = user_review.lower()
            processed_review = re.sub(r"'(s|d)\b", "", processed_review)
            processed_review = processed_review.replace("'", "")
            words = re.findall(r'\b\w+\b', processed_review)
            for word in words:
                if should_count_word(word):
                    word_count[word] += 1
            save_word_count(word_count)

        # Always show balloons and thanks message, even if review isn't saved
        st.balloons()
        st.success("Review submitted! Thanks for sharing your thoughts!")
        st.rerun()
    else:
        st.warning("Please write something before submitting.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div style="background-color: #111827; padding: 0px 0; max-width: 100%; margin-bottom: 0;">
    <div style="text-align: center; color: gray;">
        © 2025 Shail K Patel · Crafted out of boredom.
    </div>
    <div style="text-align: center; color: gray;">
        <a href="https://github.com/ShailKPatel/Falsifi.AI/" style="color: gray; text-decoration: none;">GitHub Repo</a> · MIT License
    </div>
</div>
""", unsafe_allow_html=True)