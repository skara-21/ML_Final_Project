# Walmart Store Sales Forecasting - დროითი მწკრივების პროგნოზირება

## პროექტის აღწერა

ეს პროექტი ეხება **Walmart Recruiting - Store Sales Forecasting** Kaggle კონკურსს, სადაც მიზანია Walmart-ის 45 მაღაზიის კვირეული გაყიდვების პროგნოზირება დეპარტამენტის მიხედვით. ეს არის კლასიკური Time Series ამოცანა, რომელიც მოითხოვს სხვადასხვა არქიტექტურის მოდელების შესწავლას და შედარებას.

### მონაცემთა აღწერა
- **მიზანი**: Walmart-ის მაღაზიების კვირეული გაყიდვების პროგნოზირება დეპარტამენტის მიხედვით
- **metrics**: Weighted Mean Absolute Error (WMAE)
- **მაღაზიები**: 45 Walmart მაღაზია სხვადასხვა ზომისა და ტიპის
- **დეპარტამენტები**: განსხვავებული დეპარტამენტები თითოეულ მაღაზიაში
- **გარე ფაქტორები**: ტემპერატურა, საწვავის ფასი, მარკდაუნები, დღესასწაულები, CPI, უმუშევრობის მაჩვენებელი და ა.შ.

### გუნდი
- **კესო ჩიხლაძე**
- **სალომე ყარაულაშვილი**

### პროექტის სტრუქტურა
```
ML_Final_Project/
├── notebooks/
│   ├── model_experiment_ARIMA.ipynb
│   ├── model_experiment_SARIMAX.ipynb  
│   ├── model_experiment_Prophet.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_LightGBM.ipynb
│   ├── model_experiment_NBEATS.ipynb
│   ├── model_experiment_TFT.ipynb
│   ├── model_experiment_PatchTST.ipynb
│   ├── model_experiment_DLinear.ipynb
│   └── model_inference.ipynb
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── features.csv
│   └── samlpleSubmission.csv
├── scripts/
│   ├── data_processor.py
│   └── time_series_utils.py
├── submissions/
│   ├── lightgbm_submission.csv
│   ├── arima_submission.csv
│   ├── xgboost_submission.csv
│   ├── dlinear_submission.csv
│   ├── prophet_submission.csv
│   ├── tft_submission.csv
│   ├── sarima_submission.csv
│   └── nbeats_submission.csv
└── README.md
```

---

## მონაცემთა ანალიზი და პრეპროცესინგი

### Data Exploration and Preprocessing

##### დღესასწაულების გავლენა გაყიდვებზე
<img src="holiday_analysis.png" width="800" alt="Holiday Analysis"/>

მადლიერების დღისა და შობის დროს გაყიდვები ყველაზე მეტად იზრდება.

##### სეზონური ტენდენციები
<img src="seasonal_trends.png" width="800" alt="Seasonal Trends"/>

კვირეული და თვიური სეზონურობის პატერნები მკაფიოდ ჩანს.

##### მონაცემების სისავსე
<img src="missing_data_heatmap.png" width="700" alt="Missing Data Heatmap"/>

CPI და Unemployment ცვლადებში ყველაზე მეტი missing values-ია.

##### ეკონომიკური ინდიკატორები
<img src="economic_trends.png" width="800" alt="Economic Trends"/>

CPI, უმუშევრობა, საწვავის ფასი და ტემპერატურის ცვლილება დროში.

##### ცვლადების კორელაცია
<img src="correlation_matrix.png" width="700" alt="Correlation Matrix"/>

ცვლადების ურთიერთდამოკიდებულება და მნიშვნელოვანი კორელაციები.

---

## მოდელების არქიტექტურები და შედეგები

### 🎯 **კლასიკური სტატისტიკური Time Series მოდელები**

#### ARIMA (AutoRegressive Integrated Moving Average)
**კონცეფცია**: ARIMA წარმოადგენს კლასიკურ სტატისტიკურ მოდელს დროითი მწკრივების ანალიზისთვის. იგი 3 პარამეტრისგან შედგება:

**კერძოფ**:
- **AR(p) - AutoRegressive**: განსაზღვრავს რამდენი წინა დროითი წერტილის მნიშვნელობა გამოიყენება მიმდინარე მნიშვნელობის პროგნოზირებისთვის. მაგალითად, თუ p=2, მოდელი იყენებს წინა 2 კვირის გაყიდვების მონაცემებს.
- **I(d) - Integrated**: განსაზღვრავს differencing-ის ხარისხს სტაციონარობის მისაღწევად. d=1 ნიშნავს ერთხელ გაწარმოებას ანუ წრფივი ტრენდების მოშორებას.
- **MA(q) - Moving Average**: აღწერს წინა q შეცდომის (prediction error) გავლენას მიმდინარე პროგნოზზე.

**იმპლემენტაციის თავისებურებები**:
- ARIMA-ს არ შეუძლია გარე ცვლადების (external features) გამოყენება
- მხოლოდ target ცვლადზე ტრენინგდება
- რადგან ~3300 store-department კომბინაცია საჭიროებს ბევრ მეხსიერებას, გამოვიყენეთ store-level aggregation (ყველა მაღაზიისთვის შევკრიბეთ გაყიდვები)

**ჰიპერპარამეტრების Grid Search**:
გატესტილი კომბინაციები: (1,1,0), (0,1,1), (1,1,1), (2,1,1), (1,1,2), (2,1,2), (0,1,2), (3,1,1), (1,1,3), (4,1,1)

**საუკეთესო პარამეტრები**: (p=2, d=1, q=3)
**Kaggle Score**: **3518.758 WMAE**


#### SARIMAX (Seasonal ARIMA with eXogenous variables)
**კონცეფცია**: SARIMAX აფართოებს ARIMA მოდელს ორი მნიშვნელოვანი კომპონენტით:

**სეზონური კომპონენტი (Seasonal)**:
- მოდელს ემატება (p,d,q,s) პარამეტრები სეზონურობისთვის
- s = 52 (ერთწლიანი სეზონურობა მონაცემებში)
- p,d,q იგივე ლოგიკით მუშაობს

**გარე ცვლადები (eXogenous)**:
- საშუალებას იძლევა გამოვიყენოთ გარე ცვლადები: CPI, unemployment, temperature, fuel_price
- ეს ცვლადები მოდელში შედის როგორც დამატებითი ინფორმაცია და წვლილი შეაქვს გადაწყვეტოლების მიღებაში

**იმპლემენტაციის დეტალები**:
- გამოყენებული სეზონური პარამეტრები: (P=2, D=1, Q=3, s=52)
- ძირითადი ARIMA პარამეტრები: (p=2, d=1, q=3)
- გარე ცვლადები: CPI, unemployment, temperature, fuel_price, holiday indicators

**საუკეთესო პარამეტრები**: ARIMA(2,1,3) + Seasonal(1,2,1,52) + exog_variables
**Kaggle Score**: **3084.71 WMAE**

#### Prophet
**კონცეფცია**: Facebook Prophet არის additive model-ი, რომელიც time series-ს წარმოდგენს ოთხი მთავარი კომპონენტის ჯამად:

**მათემატიკური ფორმულა**: y(t) = g(t) + s(t) + h(t) + ε(t)

**კომპონენტების დეტალური აღწერა**:
- **g(t) - Trend**: არაწრფივი ტრენდი piecewise linear ფუნქციით
  - ავტომატურად პოულობს changepoints-ებს მონაცემებში
  - იყენებს Bayesian approach-ს trend flexibility-ისთვის
- **s(t) - Seasonality**: ფურიეს სერიების საფუძველზე
  - ყოველწლიური და კვირეული სეზონურობა
  - s(t) = Σ(aₙcos(2πnt/P) + bₙsin(2πnt/P))
- **h(t) - Holidays**: დღესასწაულების effects
  - თითოეული holiday-ისთვის ცალკე parameter
  - windows around holidays (before/after effects)
- **ε(t)**: error term

**იმპლემენტაციის ფეიჩერები**:
- ავტომატურად ხლავს missing values
- robust outliers-ისადმი
- ადვილი hyperparameter tuning
- uncertainty intervals პროგნოზებისთვის

**საუკეთესო პარამეტრები**: 
```python
seasonality_mode = 'multiplicative'  # ან 'additive'
yearly_seasonality = True
weekly_seasonality = True
daily_seasonality = False
holidays = [thanksgiving, christmas, superbowl, labor_day]
```
**Kaggle Score**: **2831 WMAE**

---

###  **Tree-Based მოდელები**

### Data Exploration and Preprocessing for Tree Models

**Tree-based მოდელებისთვის Feature Engineering**:
Tree-based მოდელები (XGBoost, LightGBM) ვერ აღიქვამენ დროს პირდაპირ, ამიტომ საჭიროა მონაცემების გარდაქმნა tabular format-ში lag features-ების და სეზონური ინდიკატორების დამატებით.

#### XGBoost (Extreme Gradient Boosting)
**კონცეფცია**: XGBoost არის ensemble method რომელიც იყენებს gradient boosting ალგორითმს:

**ალგორითმის მუშაობის პრინციპი**:
1. **Sequential Tree Building**: ხეები შეიქმნება თანმიმდევრობით
2. **Gradient Descent**: ყოველი ახალი ხე ცდილობს წინა ხეების შეცდომების შესწორებას
3. **Regularization**: L1 და L2 regularization overfitting-ის თავიდან ასაცილებლად
4. **Feature Subsampling**: ყოველ iteration-ზე მხოლოდ ნაწილობრივ features გამოიყენება

**მათემატიკური გამოხატულება**:
```
ŷᵢ = Σ(k=1 to K) fₖ(xᵢ)
Obj = Σ(i=1 to n) l(yᵢ, ŷᵢ) + Σ(k=1 to K) Ω(fₖ)
```
სადაც l არის loss function, Ω არის regularization term

**Feature Engineering დეტალები**:

**1. Lag Features**:
- **Simple Lags**: sales_lag_1, sales_lag_2, sales_lag_3, sales_lag_5, sales_lag_26, sales_lag_52
- **რატომ ეს lag-ები**: ACF/PACF ანალიზის საფუძველზე მნიშვნელოვანი autocorrelations

**2. Rolling Window Features**:
- **Rolling Mean**: sales_rolling_mean_4, sales_rolling_mean_8, sales_rolling_mean_12
- **Rolling Std**: sales_rolling_std_4, sales_rolling_std_8
- **Rolling Max/Min**: sales_rolling_max_4, sales_rolling_min_4

**3. Exponentially Weighted Features**:
- **EWM**: sales_ewm_alpha_0.3, sales_ewm_alpha_0.7
- **მიზანი**: უფრო ახალ მონაცემებს მეტი წონა

**4. Date/Time Features**:
- **Linear**: year, month, week, day_of_year, week_of_year
- **Cyclical Encoding**: 
  - month_sin = sin(2π * month / 12)
  - month_cos = cos(2π * month / 12)
  - week_sin, week_cos

**5. Economic Features**:
- **Interpolated**: CPI_interpolated, unemployment_interpolated
- **Derived**: economic_pressure = CPI * unemployment
- **Markdown**: total_markdown, markdown_effectiveness

**6. Store/Dept Statistics**:
- **Historical**: store_mean_sales, dept_mean_sales, store_dept_mean_sales
- **Variability**: store_sales_std, dept_sales_std

*
**ჰიპერპარამეტრების დეტალური აღწერა**:
- **learning_rate (0.01-0.3)**: gradient descent-ის step size
- **n_estimators (100-5000)**: ხეების რაოდენობა
- **max_depth (3-15)**: ხის მაქსიმალური სიღრმე
- **min_child_weight (1-10)**: leaf node-ის მინიმალური წონა
- **subsample (0.6-1.0)**: sample fraction ყოველ iteration-ზე
- **colsample_bytree (0.6-1.0)**: feature fraction ყოველ ხეზე
- **gamma (0-5)**: minimum split loss
- **reg_alpha (0-10)**: L1 regularization
- **reg_lambda (0-10)**: L2 regularization


**Kaggle Score**: **4544 WMAE**

#### LightGBM (Light Gradient Boosting Machine)
**კონცეფცია**: LightGBM არის Microsoft-ის მიერ შექმნილი gradient boosting framework, რომელიც ოპტიმიზებულია სიჩქარისა და მეხსიერების ეფექტურობისთვის.

**Tree Growing Strategy**:
- **LightGBM**: Leaf-wise tree growth (ყველაზე დიდი loss reduction-ის მქონე ფოთოლი)

**მეხსიერების გამოყენება**:
- **Feature Bundling**: sparse features-ების გაერთიანება
- **Histogram-based Algorithm**: რიცხვების binning მეხსიერების დასაზოგად

**Categorical Features**:
- განიხილავს categorical variables-ებს encoding-ის გარეშე
- ავტომატური optimal split-ების პოვნა

**იმპლემენტაციის თავისებურებები**:
- იგივე feature engineering როგორც XGBoost-ისთვის
- უფრო მცირე hyperparameter space (ჩქარია training)
- categorical features-ების encoding გარეშე უკეთესი შედეგები




#### DLinear (Decomposition Linear)
**კონცეფცია**: DLinear არის გამოცდილად მარტივი მოდელი, რომელიც აჩვენებს რომ კომპლექსური neural networks ყოველთვის არ არის საჭირო time series forecasting-ისთვის.

**არქიტექტურის სიმარტივე**:
```
Input Time Series → Decomposition → Linear Layers → Output Forecast
```

**Decomposition მექანიზმი**:
1. **Seasonal Decomposition**:
   ```
   Seasonal = MovingAverage(Input, kernel_size)
   ```
2. **Trend Decomposition**:
   ```
   Trend = Input - Seasonal
   ```

**Linear Forecasting**:
```
Seasonal_Forecast = Linear_Seasonal(Seasonal_Component)
Trend_Forecast = Linear_Trend(Trend_Component)
Final_Forecast = Seasonal_Forecast + Trend_Forecast
```

**მარტივი მაგრამ ეფექტური**:
- მხოლოდ 2 linear layer
- არ საჭიროებს რთულ architecture-ს
- ძალიან სწრაფი training და inference
- მაგრამ ხშირად კონკურენტუნარიანია complex models-თან

**ჰიპერპარამეტრების მინიმალობა**:
- **seq_len**: input sequence length
- **pred_len**: prediction horizon
- **kernel_size**: moving average window size decomposition-ისთვის
- **individual**: თითოეული channel-ისთვის ცალკე parameters



**Kaggle Score**: **4617 WMAE**

#### N-BEATS (Neural Basis Expansion Analysis for Time Series)
**კონცეფცია**: N-BEATS არის pure deep learning მოდელი time series forecasting-ისთვის, რომელიც არ საჭიროებს domain knowledge-ს ან feature engineering-ს.

**არქიტექტურის დეტალური აღწერა**:

**1. Stack და Block სტრუქტურა**:
```
N-BEATS = Stack₁ + Stack₂ + ... + Stackₙ
Stack = Block₁ + Block₂ + ... + Blockₘ
```

**2. Block Types**:
- **Trend Block**: პოლინომიური ბაზის ფუნქციები
  - Basis Functions: [1, t, t², t³, ..., tᵖ]
  - პარამეტრი: polynomial_degree
- **Seasonality Block**: ფურიეს ბაზის ფუნქციები  
  - Basis Functions: [1, cos(2πt), sin(2πt), cos(4πt), sin(4πt), ...]
  - პარამეტრი: num_harmonics
- **Generic Block**: ზოგადი ნეირონული ქსელი
  - სრული flexibility residuals-ისთვის

**3. Forecasting მექანიზმი**:
ყოველი Block-ისთვის ხდება:
- **Backcast (bₖ)**: input-ის reconstruction
- **Forecast (fₖ)**: მომავლის prediction

**ჰიპერპარამეტრების დეტალური აღწერა**:
- **input_size (lookback)**: რამდენით წინა დროის მონაცემს გამოიყენებს
- **output_size (horizon)**: რამდენით მომავალი დროის მონაცემს დააპროგნოზირებს
- **stack_types**: ['trend', 'seasonality', 'generic'] კომბინაცია
- **nb_blocks_per_stack**: ბლოკების რაოდენობა თითოეულ stack-ში
- **thetas_dim**: internal representation dimension
- **share_weights_in_stack**: weights sharing მექანიზმი
- **hidden_layer_units**: FC layers-ის ზომა

**არქიტექტურის თავისებურებები**:
- თითოეული ბლოკი აკეთებს forecast-ს  და backcast-ს
- ბლოკები ერთიანდება stack-ებში საბოლოო forecast-ისთვის
- არ საჭიროებს feature engineering-ს

**Kaggle Score**: **3100.07**


#### PatchTST (Patching Time Series Transformer)
**კონცეფცია**: PatchTST არის ახალი transformer-based approach რომელიც იყენებს "patching" ტექნიკას time series-ის უკეთესი modeling-ისთვის.

**Patching მექანიზმი**:
```
Time Series: [x₁, x₂, x₃, ..., x₁₀₀]
მაგ: Patching (patch_size=10, stride=5)
Patches: [[x₁...x₁₀], [x₆...x₁₅], [x₁₁...x₂₀], ...]
```

**არქიტექტურის ნაბიჯები**:

**1. Patching და Linear Embedding**:
```
Input Time Series → Patch Creation → Linear Projection → Patch Embeddings
Positional Encoding → Final Input Embeddings
```

**2. Transformer Encoder**:
```
Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm
```

**3. Forecasting Head**:
```
Patch Embeddings → Linear Layers → Future Values
```

**Channel Independence**:
- ყოველი time series (channel) იმუშავებს დამოუკიდებლად
- არ არის cross-channel information sharing
- უკეთესი განზოგადება
- უფრო stable training უცნობ patterns-ზე

**მთავარი იდეა - Patching**:
PatchTST approach:
- თითოეული patch = ერთი token
- მოკლე sequence of patches → დაბალი კომპლექსურობა
- ადგილობრივი პატერნები patch-ებში + გლობალური პატერნები patch-ს შორის


**ჰიპერპარამეტრების მნიშვნელობა**:
- **seq_len**: input sequence-ის სიგრძე
- **pred_len**: prediction horizon
- **patch_len**: ყოველი patch-ის სიგრძე (მნიშვნელოვანი!)
- **stride**: patches-ს შორის დაშორება
- **d_model**: embedding dimension
- **n_heads**: attention heads-ის რაოდენობა
- **d_ff**: feed-forward dimension
- **n_layers**: transformer layers-ის რაოდენობა
- **dropout**: regularization parameters

**Channel Independence უპირატესობები**:
- თითოეული time series სწავლობს საკუთარ patterns
- უკეთესი scalability დიდ dataset-ებზე
- უფრო მრავალფეროვანი


### Model Pipeline და Registry

**Pipeline სტრუქტურა**:
ყოველი საუკეთესო მოდელი შენახულია Pipeline ფორმატში:

pipeline = Pipeline([
    ('preprocessor', custom_preprocessor),
    ('feature_engineer', feature_transformer),
    ('model', best_model)
])

# Model Registry-ში რეგისტრაცია
mlflow.sklearn.log_model(
    pipeline, 
    "model",
    registered_model_name=f"{model_name}_walmart_forecasting"
)
- ეს მხოლოდ sklearn მოდელებს

**Pipeline კომპონენტები**:
- **Data Validation**: მონაცემების ხარისხის შემოწმება
- **Preprocessing**: missing values handling, outlier detection
- **Feature Engineering**: lag features, rolling statistics, date features
- **Model**: trained forecasting model
- **Postprocessing**: predictions formatting, confidence intervals

---

## მოდელების შედარება და საბოლოო შედეგები


**საბოლოო შედეგების ცხრილი:**


### Performance Analysis

**ტოპ 3 მოდელი**:
1. 🥇 **Prophet**
2. 🥈 **SARIMA**
3. 🥉 **N-BEATS**

**მოდელების კატეგორიების მიხედვით ანალიზი**:

**Statistical Models (ARIMA, SARIMAX, Prophet)**:
- **უპირატესობები**: სწრაფი, ინტერპრეტირებადი, მცირე რესურსები
- **ნაკლოვანებები**: ლიმიტირებული feature capacity, სუსტი multivariate handling

**Tree-Based Models (XGBoost, LightGBM)**:
- **უპირატესობები**: მძლავრი feature engineering, კარგი performance/effort ratio
- **ნაკლოვანებები**: manual feature engineering, დროითი პატერნები ხელით უნდა შექმნა

**Deep Learning Models (N-BEATS, TFT, PatchTST, DLinear)**:
- **უპირატესობები**: ავტომატური pattern recognition, long-term dependencies
- **ნაკლოვანებები**: მძმე გამოთვლები, hyperparameter მგრძნობელობა


### Time Series Forecasting-ის სპეციფიკური გაკვეთილები

**1. Data Leakage-ის თავიდან აცილება**:
- Temporal validation splits-ის მნიშვნელობა
- Feature engineering და temporal consistency

**2. Model Selection Criteria**:
- უნდა ვიპოვოთ ბალანსი კომპიუტერულ რესურსებს, დროსა და შედეგბს შორის

### მომავალი გაუმჯობესებები

**Short-term Improvements**:
- ვეცდებოდით deep learning მოდელების უკეთესი პარამეტრებით გაწვრთნას

**ტექნიკური შეზღუდვები რაც გქონდათ**:
- ვმუშაობდით ჩვეულებრივ Collab-ში და შეზღუდული იყო GPU usage
- ცხელა

---