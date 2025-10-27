# 📰 News Recommendation System using MIND Dataset and Sentence-BERT

## 🧩 Problem Statement
This project implements a **personalized news recommendation system** using the **MIND (Microsoft News Dataset)**.  
The goal is to recommend relevant news articles to users based on their past interactions (clicks/impressions) by leveraging **Sentence-BERT embeddings** and **similarity-based retrieval**.

---

## 📊 Dataset Details

### Dataset Source
- **Dataset:** [MINDsmall_train (Kaggle)](https://www.kaggle.com/datasets)
- **Files Used:**
  - `news.tsv` → Contains article metadata (ID, category, subcategory, title, abstract, URL, entities)
  - `behaviors.tsv` → Contains user interaction logs (impressions, clicked/not-clicked data)

### Data Statistics
- **News Articles:** ~51,282 entries  
- **Users:** ~94,000 unique user IDs  
- **Impressions:** Several hundred thousand logs  
- **Fields:**
  - `news_id`, `category`, `subcategory`, `title`, `abstract`, `entities`
  - `user_id`, `history`, `impressions` (clicked / not-clicked)

---

## ⚙️ Methodology

### 1. Environment & Model Setup
- Initialized environment on Kaggle runtime.  
- Cloned **Hugging Face SentenceTransformer model:**  
  **`all-MiniLM-L6-v2`** (384-dimensional embeddings).  
- Libraries used:  
  `pandas`, `numpy`, `sentence-transformers`, `scikit-learn`, `time`.

### 2. Data Loading
```python
news = pd.read_csv("/kaggle/input/mind-news-dataset/MINDsmall_train/news.tsv", sep='\t')
behavior = pd.read_csv("/kaggle/input/mind-news-dataset/MINDsmall_train/behaviors.tsv", sep='\t')
```
- Loaded **51,282 news articles** and **behavior logs**.  
- Verified **18 main categories** and **more than 250 subcategories**.

### 3. Parsing User Impressions
Each user’s impressions were parsed from the format `N12345-1 N98765-0` into structured tuples of `(news_id, click)`:
```python
def parse_impressions(row):
    pairs = row.split()
    return [(nid, int(click)) for nid, click in [p.split('-') for p in pairs]]
```
- Extracted clicked (positive) and non-clicked (negative) samples.  
- Computed **Click-Through Rate (CTR): 0.95%** for example user.

### 4. Text Preparation
- Combined **title** and **abstract** for each article → average length: 291 characters.  
- Tokenization and preprocessing done inline prior to embedding.

### 5. Embedding Generation
Used **Sentence-BERT (all-MiniLM-L6-v2)** for semantic encoding.
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(news['title'] + " " + news['abstract'])
```
- **Embedding Shape:** (51,282, 384)  
- **Time Taken:** ~45.3 seconds (~0.88 ms/article)

### 6. Building Recommendation Index
- Constructed **K-Nearest Neighbors (KNN)** index using cosine similarity.  
- Computed user preference vectors:
  - Positive matrix: `(3, 384)`  
  - Negative matrix: `(313, 384)`  
- Derived **centroid representation** for both positive and negative preferences.

### 7. Recommendation Computation
- Calculated cosine distances between user preference centroid and all news embeddings.  
- Ranked closest 3–5 items using **`NearestNeighbors(metric='cosine')`**.

---

## 🧠 Results

### Execution Summary
| Step | Description | Time (s) |
|------|--------------|----------|
| Data Loading | Read news & behavior files | 5.2 |
| Impression Parsing | Click log parsing | 2.1 |
| Embedding Generation | Sentence-BERT (MiniLM-L6) | 45.3 |
| Similarity Search | Cosine distance computation | 0.01 |
| **Total** |  | **≈ 52.6 seconds** |

### Embedding Insights
| Metric | Value |
|--------|-------|
| Min cosine distance | 0.8048 |
| Max cosine distance | 0.9969 |
| Mean cosine distance | 0.8787 |

### Example: Top 5 Recommendations (User: U13740)
| Rank | Category | Subcategory | Title |
|------|-----------|--------------|--------|
| 1 | travel | traveltips | *What Happens If Your Oxygen Mask Doesn't Inflate on a Flight?* |
| 2 | sports | football_nfl | *Charles Rogers, former Michigan State football, Detroit Lions star, dead at 38* |
| 3 | news | newspolitics | *George Kent, top State Department Ukraine expert, helps Democrats debunk GOP theories* |

✅ **CTR Analyzed:** 0.95%  
✅ **Model:** SentenceTransformer (`all-MiniLM-L6-v2`)  
✅ **Embedding Dimension:** 384  
✅ **KNN Index:** cosine similarity-based (k=3)

---

## 📈 Conclusion
- The **Sentence-BERT embedding–based recommendation system** accurately captured semantic similarity across diverse categories.  
- Generated embeddings enable fast, contextual matching between unseen news articles and user preferences.  
- Execution efficiency: full run under **53 seconds** on Kaggle GPU/TPU runtime.  
- Highly adaptable to **personalized recommendations** for multiple users simultaneously.

---

## 🧰 Technologies Used
- **Python 3.11**  
- **Pandas**, **NumPy**, **scikit-learn**  
- **Sentence-Transformers (BERT)**  
- **Kaggle Notebook Environment**  
- **Cosine Similarity & KNN-based retrieval**

---

## ✨ Author
Developed by **Akshat**  
> “Learning what the user truly wants — one embedding at a time.”
