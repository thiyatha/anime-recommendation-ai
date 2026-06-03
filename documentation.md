# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.


## Project Metadata

- Project title:  Anime Recommendation AI
- Student: Thajeena Thiyagarajah
- GitHub repository URL: https://github.com/thiyatha/anime-recommendation-ai
- Deployment URL: https://huggingface.co/spaces/thajee/anime-recommendation-ai
- Submission date: 07.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: NLP
- Primary block 2: ML Numeric Data

If a third block is selected, it is documented and graded separately as extra work.

Third block selected:
- Computer Vision


---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement:  
  Anime fans often receive generic recommendations that are mainly based on popularity or manual genre filters. These approaches do not fully consider natural language preferences, numeric anime metadata, and uploaded visual references such as anime screenshots or posters.

- Goal:  
  The goal of this project is to build a multimodal anime recommendation system that combines user text, structured anime metadata, and optional image input to recommend suitable anime titles.

- Success criteria:  
  The system should:
  - accept a natural language anime preference,
  - use OpenAI and classical NLP methods to understand the user request,
  - compare at least two numeric ML models for anime score prediction,
  - use CLIP-based Computer Vision to match uploaded anime images,
  - combine all outputs into a final recommendation score,
  - run as a working Hugging Face Space deployment.

### 1.2 Integration Logic
- How the selected blocks interact:  
  The app combines NLP, Numeric ML, and Computer Vision into one recommendation pipeline.

  1. The NLP block analyzes the user input using Keyword Matching, TF-IDF similarity, and OpenAI structured preference extraction.
  2. The Numeric ML block predicts anime scores using structured features such as episodes, members, and mood levels.
  3. The Jikan API is used as an external data source to enrich the recommendation output with additional anime metadata.
  4. The Computer Vision block is optional and uses CLIP to compare an uploaded image with anime titles and metadata. If no image is uploaded, the visual score remains neutral.
  5. The final recommendation score combines NLP score, mood score, ML numeric score, and CLIP visual score.

- Data and output flow between blocks:
```text
User text input
    ↓
NLP block
Keyword Matching + TF-IDF + OpenAI preference extraction
    ↓
Mood preferences and NLP similarity scores
    ↓
Local anime dataset + Jikan API metadata
    ↓
Numeric ML block
Linear Regression + Random Forest prediction
    ↓
Predicted anime score
    ↓
Optional image upload
    ↓
Computer Vision block
CLIP image-to-text similarity
    ↓
Visual anime match score
    ↓
Final weighted recommendation score
    ↓
Top 5 anime recommendations
```

---

## 2. Block Documentation


### 2A. ML Numeric Data

#### 2A.1 Data Source(s)
| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | `data/anime_data.csv` | Structured CSV dataset | 80 anime entries | Main numeric dataset for training and prediction |
| 2 | Engineered mood features in `anime_data.csv` | Numeric feature columns | 4 mood columns per anime | Used as model input features |
| 3 | User interaction data during inference | Runtime input | One request per user interaction | Used indirectly for final scoring and recommendation context |

The dataset was manually created for this project and contains known anime titles with structured metadata. It does not use the semester apartment dataset or dog image dataset.


#### 2A.2 Preprocessing and Features
- Cleaning steps:
  - The CSV file was checked for consistent column names.
  - Numeric columns such as `episodes`, `score`, `members`, `dark_tone`, `romance_level`, `action_level`, and `emotional_level` are loaded as numeric values.
  - Text columns such as `title`, `genres`, `synopsis`, and `image_style` are kept for NLP and CV prompt construction.

- Preprocessing steps:
  - The dataset is loaded with Pandas in [`app.py`](app.py).
  - The ML features are selected in [`ml_model.py`](ml_model.py).
  - The data is split into training and test data using `train_test_split`.

- Feature engineering and selection:
  - Selected numeric features:
    - `episodes`
    - `members`
    - `dark_tone`
    - `romance_level`
    - `action_level`
    - `emotional_level`
  - Target variable:
    - `score`

These features were selected because they represent both objective metadata and engineered mood indicators.


#### 2A.3 Model Selection
- Models tested:
  - Linear Regression
  - Random Forest Regression

- Why these models were chosen:
  - Linear Regression was selected as a simple baseline model.
  - Random Forest Regression was selected because it can model non-linear relationships between metadata features and anime scores.
  - Comparing both models gives a clear evaluation between a simple model and a stronger ensemble model.


#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Manual numeric scoring | Used formula based on rating, members, and episodes | No trained ML model | Qualitative check | First baseline |
| 2 | Add model training | Added Linear Regression and Random Forest | Linear Regression, Random Forest | MAE, RMSE, R2 | Replaced formula with trained models |
| 3 | Separate ML module | Moved ML logic into `ml_model.py` | Linear Regression, Random Forest | MAE, RMSE, R2 | Cleaner structure and easier documentation |


#### 2A.5 Evaluation and Error Analysis
- Metrics used:
  - MAE: Mean Absolute Error
  - RMSE: Root Mean Squared Error
  - R2 Score

- Final results from the current app run:
  - Best numeric ML model: Random Forest
  - Linear Regression:
    - MAE: 0.267
    - RMSE: 0.373
    - R2: 0.137
  - Random Forest:
    - MAE: 0.238
    - RMSE: 0.352
    - R2: 0.231

- Error patterns and likely causes:
  - The R2 values are relatively low because the dataset is small and anime scores are subjective.
  - Popularity, episodes, and mood features cannot fully explain user ratings.
  - Some anime with niche audiences may have high quality but lower member counts.
  - Some mainstream anime may have high popularity but not always the highest rating.


#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - The final recommendation uses NLP and CV scores together with the numeric ML score.
  - The mood features used in ML are also connected to the NLP preference extraction.

- Outputs provided to other block(s):
  - The ML block outputs a predicted anime score.
  - This score is normalized and used as `ML Numeric Score`.
  - The final app combines it with NLP, mood, and CLIP visual scores.


### 2B. NLP 

#### 2B.1 Data Source(s)
| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | User text input | Natural language text | One prompt per request | Main input for preference extraction |
| 2 | `data/anime_data.csv` | Text metadata | 80 anime descriptions | Used for matching genres, synopsis, and image style |
| 3 | OpenAI API | LLM-based structured extraction | One API call per request when key is available | Extracts mood levels, intent, and preferred genres |


#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing:
  - User input is converted to lowercase.
  - Basic punctuation such as commas, dots, question marks, exclamation marks, and hyphens is removed.
  - Common stopwords such as `i`, `want`, `a`, `the`, `with`, `give`, `me`, and `anime` are removed for keyword matching.
  - For each anime, one combined text representation is created from:
    - `genres`
    - `synopsis`
    - `image_style`

- Prompt design:
  - OpenAI is used to extract structured anime preferences from the user prompt.
  - The prompt asks the model to return values that can be used by the recommendation system.
  - The returned structure contains:
    - `dark_tone`
    - `romance_level`
    - `action_level`
    - `emotional_level`
    - `preferred_genres`
    - `intent`
    - `short_explanation`


#### 2B.3 Approach Selection
- Approach used:
  - Keyword Matching
  - TF-IDF Cosine Similarity
  - OpenAI structured mood and intent extraction

- Why these approaches were chosen:
  - Keyword Matching was chosen as a simple and transparent baseline.
  - TF-IDF was chosen as a stronger classical NLP method because it compares the user prompt with all anime texts.
  - OpenAI was added to better understand user intent and extract structured preferences from natural language.

- Alternatives considered:
  - Only using keyword matching was too limited because it does not understand context well.
  - Only using TF-IDF was better than simple keyword matching, but still weak for intent recognition.
  - Using only OpenAI would reduce transparency, so it was combined with classical NLP methods.

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Basic text matching | Keyword matching against genres and synopsis | Rule-based keyword method | Qualitative result check | First NLP baseline |
| 2 | Improve similarity | Added TF-IDF Cosine Similarity | `TfidfVectorizer` | Qualitative comparison of recommendations | More flexible text similarity |
| 3 | Improve intent understanding | Added OpenAI structured extraction | OpenAI JSON-style preference extraction | Checks extracted intent and genres | Better natural language understanding |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy:
  - Several representative user prompts were tested manually.
  - The recommendation results were checked qualitatively.
  - The OpenAI output was inspected in the app output.
  - The extracted intent, preferred genres, and explanation were compared with the original user prompt.

- Results:
  - Prompt: `i want a funny anime`
    - OpenAI used: True
    - Intent: recommendation
    - Preferred genres: `['comedy']`
    - Explanation: User expressed a desire for a funny anime, indicating a preference for comedy.
    - Result: The recommendations included comedy and slice-of-life anime such as Barakamon.
  - Prompt: `give me a romance anime`
    - Result: The recommendations included romance titles such as A Silent Voice, Your Name, Spice and Wolf, Honey and Clover, and Nana.
  - Prompt: `I want a dark emotional anime with action and strong character development`
    - Result: The recommendations included dark, action, and drama anime such as Attack on Titan, Solo Leveling, Naruto Shippuden, Jujutsu Kaisen, and Neon Genesis Evangelion.

- Error patterns and likely causes:
  - Keyword Matching can fail if the user uses synonyms that do not appear in the dataset.
  - TF-IDF can give high weight to rare words even if the semantic meaning is not very strong.
  - OpenAI improves intent understanding, but it depends on the API key and model availability.
  - If OpenAI is unavailable, the app uses a rule-based fallback.
  - Short prompts such as `romance` or `funny` can work, but longer prompts usually provide better context.


#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - The NLP block uses anime text metadata from `data/anime_data.csv`.
  - The OpenAI intent detection helps decide whether the user wants a normal recommendation or image identification.

- Outputs provided to other block(s):
  - Keyword Score
  - TF-IDF Score
  - OpenAI extracted mood values
  - Preferred genres
  - Intent, for example `recommendation` or `image_identification`

These outputs are used in the final weighted recommendation score.  
The NLP output is combined with the Numeric ML score and the CLIP Visual Score to produce the final Top 5 anime recommendations.

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | User uploaded image | Image input | One image per request | Main Computer Vision input |
| 2 | `data/anime_data.csv` | Text metadata | 80 anime titles and styles | Used to create CLIP text prompts |
| 3 | CLIP model `openai/clip-vit-base-patch32` via Hugging Face Transformers | Pretrained vision-language model | Pretrained model | Used for image-to-text similarity |


#### 2C.2 Preprocessing and Augmentation
- Image preprocessing:
  - Uploaded images are converted to RGB format.
  - The CLIP processor handles resizing, normalization, and tensor conversion.
  - Text prompts are generated from anime title, genres, and image style.
  - Each uploaded image is compared against all anime entries in the dataset.

- Augmentation strategy:
  - No custom image augmentation was used.
  - The app performs zero-shot image matching during inference.
  - The focus is not on training a custom image classifier, but on applying a pretrained vision-language model.

#### 2C.3 Model Selection
- Vision model used:
  - CLIP: `openai/clip-vit-base-patch32`

- Why this model was chosen:
  - CLIP can compare images and text in a shared embedding space.
  - This fits the use case because the user uploads an anime image and the app compares it to text prompts such as anime titles, genres, and visual styles.
  - CLIP supports zero-shot image matching without requiring a large custom image dataset.
  - This makes it suitable for a small project with limited training data.

#### 2C.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Basic visual mood analysis | Planned brightness and contrast analysis | Simple image statistics | Visual inspection | First CV idea |
| 2 | Anime image matching | Added CLIP image-to-text similarity | CLIP | Visual match confidence | Stronger CV functionality |
| 3 | Better integration | Increased CLIP weighting for image identification prompts | CLIP + intent detection | Qualitative check | Better ranking for image identification |

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks:
  - CLIP visual match confidence
  - Qualitative check of closest visual anime match
  - Manual testing with uploaded anime images

- Final results:
  - Test image: Naruto image
  - User prompt: `what anime is this`
  - Closest visual anime match: Naruto
  - Visual match confidence: 0.646
  - The app correctly identified Naruto as the closest visual match.

- Error patterns and limitations:
  - CLIP is not a specialized anime character recognition model.
  - It may confuse visually similar anime styles.
  - The model can only choose from anime titles included in `anime_data.csv`.
  - If an anime is not included in the dataset, it cannot be returned as the closest match.
  - Low-quality, cropped, or unclear images may reduce accuracy.
  - Posters usually work better than very small screenshots.


#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s):
  - The NLP block provides the detected user intent.
  - If the user asks something like `what anime is this`, the app switches to image identification mode.
  - Anime metadata from `data/anime_data.csv` is used to create text prompts for CLIP.

- Outputs provided to other block(s):
  - CLIP Visual Score for every anime
  - Closest visual anime match
  - Visual match confidence

The CV score is included in the final recommendation score.  
If the user asks `what anime is this`, the CLIP visual score receives a higher weight so that the uploaded image becomes more important than the normal recommendation logic.

---

## 3. Deployment

The app is deployed on Hugging Face Spaces using Gradio.
The main file is app.py.
The required packages are listed in requirements.txt.

- Main user flow:
  1. User opens the Hugging Face Space.
  2. User enters a natural language anime preference.
  3. User optionally uploads an anime image or poster.
  4. The app calculates NLP, ML, and CV scores.
  5. The app returns the Top 5 anime recommendations and an explanation.

- Screenshot or short demo:

Screenshot 1: OpenAI text recommendation example  
- Input: `i want a funny anime`
- Shows:
  - OpenAI used: True
  - Preferred genres: `['comedy']`
  - Top anime recommendations

File reference:
- `screenshots/openai_comedy_test.png`

Screenshot 2: Romance recommendation example  
- Input: `give me a romance anime`
- Shows:
  - Romance-related recommendations
  - NLP and ML evaluation

File reference:
- `screenshots/romance_recommendation_test.png`

Screenshot 3: Computer Vision image identification example  
- Input: `what anime is this`
- Uploaded image: Naruto image
- Shows:
  - Task mode: Image identification
  - Closest visual anime match: Naruto
  - Visual match confidence: 0.646

File reference:
- `screenshots/naruto_clip_test.png`
---

## 4. Execution Instructions

- Environment setup:
```bash
git clone [(https://github.com/thiyatha/anime-recommendation-ai)]
cd anime-recommendation-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Data setup: data/anime_data.csv

- Training command(s):
No separate training command is required.
The numeric ML models are trained automatically when the app starts.  
The training logic is implemented in:
```text
ml_model.py
```
- Inference/run command(s): ```python app.py```

- Reproducibility notes:
The dataset is stored directly in the repository.
The dataset path is ```data/anime_data.csv```.
The train/test split uses random_state=42.
The Random Forest model uses random_state=42.
CLIP is loaded through Hugging Face Transformers.
OpenAI extraction depends on the configured API key and selected model.
If OpenAI is unavailable, the app still runs with the rule-based fallback.
The deployed version runs on Hugging Face Spaces.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case

Evidence for selected bonus items:
