import gradio as gr
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import pickle

vectorizer_teacher_prefix = pickle.load(open("/email/dsml/models/vectorizer_teacher_prefix.pkl", "rb"))
vectorizer_school_state = pickle.load(open("/email/dsml/models/vectorizer_school_state.pkl", "rb"))
vectorizer_sub = pickle.load(open("/email/dsml/models/vectorizer_subject_Category.pkl", "rb"))
vectorizer_sub_sub_category = pickle.load(open("/email/dsml/models/vectorizer_subject_subcategory.pkl", "rb"))
vectorizer_project_grade_category = pickle.load(open("/email/dsml/models/vectorizer_project_grade_category.pkl", "rb"))
vectorizer_project_title_category = pickle.load(open("/email/dsml/models/vectorizer_project_title_category.pkl", "rb"))
vectorizer_resource_summary = pickle.load( open("/email/dsml/models/vectorizer_resource_summary.pkl", "rb"))
vectorizer_project_essay_category = pickle.load(open("/email/dsml/models/vectorizer_project_essay_category.pkl", "rb"))
normalizer = pickle.load(open("/email/dsml/models/normalizer.pkl", "rb"))
mnb_bow_testModel = pickle.load(open("/email/dsml/models/mnb_bow_testModel.pkl", "rb"))

# If facing pickle error while loading the model use following command
# file_path="/email/dsml/models/mnb_bow_testModel.pkl"

# try:
#     with open(file_path, "rb") as file:
#         try:
#             mnb_bow_testModel = pickle.load(file)
#             # Process the unpickled data here
#             print("Data successfully unpickled:", mnb_bow_testModel)
#         except EOFError:
#             print("Error: The file is empty or contains no valid pickled data.")
#         except Exception as e:
#             print("An error occurred while unpickling:", str(e))
# except FileNotFoundError:
#     print("Error: File not found at", file_path)
# except Exception as e:
#     print("An error occurred while opening the file:", str(e))







def predict(teacher_prefix, school_state, project_subject_categories, project_subject_subcategories, project_grade_category, project_title,project_resource_summary,essay,teacher_number_of_previously_posted_projects, price,quantity):
    # Process the inputs and generate the output
    # You can replace this with your own logic or model
    # Working on  vectorizer_subject_Category

    # Vectorizing teacher_prefix
    # vectorizer_teacher_prefix = pickle.load(open("/email/dsml/models/vectorizer_teacher_prefix.pkl", "rb"))
    x_test_teacher_prefix_one_hot  = vectorizer_teacher_prefix.transform([teacher_prefix])
    print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_teacher_prefix_one_hot.shape)

    # Vectorizing school_state
    # vectorizer_school_state = pickle.load(open("/email/dsml/models/vectorizer_school_state.pkl", "rb"))
    x_test_school_state_one_hot  = vectorizer_school_state.transform([school_state])
    print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_school_state_one_hot.shape)


    # Working on  vectorizer_subject_Category
    # vectorizer_sub = pickle.load(open("/email/dsml/models/vectorizer_subject_Category.pkl", "rb"))    
    x_test_project_subject_categories_one_hot  = vectorizer_sub.transform([project_subject_categories])
    print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_subject_categories_one_hot.shape)

    # Vectorizing project subject sub-categories
    # vectorizer_sub_sub_category = pickle.load(open("/email/dsml/models/vectorizer_subject_subcategory.pkl", "rb"))
    x_test_project_subject_subcategories_one_hot  = vectorizer_sub_sub_category.transform([project_subject_subcategories])
    print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_subject_subcategories_one_hot.shape)

    # Vectorizing project_grade_category
    # vectorizer_project_grade_category = pickle.load(open("/email/dsml/models/vectorizer_project_grade_category.pkl", "rb"))
    x_test_project_grade_category_one_hot  = vectorizer_project_grade_category.transform([project_grade_category])
    print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_grade_category_one_hot.shape)

    # Applying CountVectorizer on project_title
    # vectorizer_project_title_category = pickle.load(open("/email/dsml/models/vectorizer_project_title_category.pkl", "rb"))
    x_test_project_titles_tfidf  = vectorizer_project_title_category.transform([project_title])
    print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_project_titles_tfidf.shape)


    # Vectorizing project project_resource_summary categories
    # vectorizer_resource_summary = pickle.load( open("/email/dsml/models/vectorizer_resource_summary.pkl", "rb"))
    x_test_project_resource_summary_one_hot  = vectorizer_resource_summary.transform([project_resource_summary])
    print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_resource_summary_one_hot.shape)

    # Applying CountVectorizer on essay
    # vectorizer_project_essay_category = pickle.load(open("/email/dsml/models/vectorizer_project_essay_category.pkl", "rb"))
    x_test_essay_tfidf  = vectorizer_project_essay_category.transform([essay])
    print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_essay_tfidf.shape)

    df=pd.DataFrame()
    # applying StandardScaler to specific columns    
    # normalizer = pickle.load(open("/email/dsml/models/normalizer.pkl", "rb"))
    # x_test[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(['teacher_number_of_previously_posted_projects', 'price',  'quantity'])
    
    df[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform([[teacher_number_of_previously_posted_projects, price,  quantity]])


        # merging test value
    x_test_onehot = hstack((x_test_project_subject_categories_one_hot,
                            x_test_project_resource_summary_one_hot,
                            x_test_project_subject_subcategories_one_hot   ,
                            x_test_teacher_prefix_one_hot    ,
                            x_test_school_state_one_hot  ,
                            x_test_project_grade_category_one_hot  ,
                            x_test_project_titles_tfidf  ,
                            x_test_essay_tfidf,
                            df[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]))


    y_test_pred=mnb_bow_testModel.predict_proba(x_test_onehot)[:,1]
    output="Project not approved"
    print("y_test= ",y_test_pred)
    # for i in y_test_pred:
    if y_test_pred>=0.5:
        output="Project approved"
    
    return output

# Define the inputs and outputs for the Gradio interface
inputs = [
    gr.inputs.Dropdown(['mrs', 'mr', 'ms', 'teacher', 'dr'], label="teacher_prefix"),
    gr.inputs.Dropdown(['in', 'fl', 'az', 'ky', 'tx', 'ct', 'ga', 'sc', 'nc', 'ca', 'ny',
       'ok', 'ma', 'nv', 'oh', 'pa', 'al', 'la', 'va', 'ar', 'wa', 'wv',
       'id', 'tn', 'ms', 'co', 'ut', 'il', 'mi', 'hi', 'ia', 'ri', 'nj',
       'mo', 'de', 'mn', 'me', 'wy', 'nd', 'or', 'ak', 'md', 'wi', 'sd',
       'ne', 'nm', 'dc', 'ks', 'mt', 'nh', 'vt'], label="school_state"),
    gr.inputs.Dropdown(['literacy_language', 'history_civics health_sports',
       'health_sports', 'literacy_language math_science', 'math_science',
       'literacy_language special_needs',
       'literacy_language applied_learning', 'special_needs',
       'math_science literacy_language', 'applied_learning',
       'math_science special_needs', 'music_the_arts', 'history_civics',
       'health_sports literacy_language',
       'literacy_language music_the_arts', 'warmth care_hunger',
       'math_science history_civics',
       'applied_learning literacy_language',
       'applied_learning special_needs',
       'literacy_language history_civics',
       'applied_learning health_sports',
       'history_civics literacy_language', 'health_sports special_needs',
       'applied_learning math_science', 'math_science music_the_arts',
       'health_sports applied_learning', 'history_civics music_the_arts',
       'math_science applied_learning', 'music_the_arts history_civics',
       'applied_learning music_the_arts', 'history_civics math_science',
       'music_the_arts applied_learning', 'health_sports music_the_arts',
       'math_science health_sports', 'special_needs health_sports',
       'health_sports math_science', 'special_needs music_the_arts',
       'music_the_arts warmth care_hunger',
       'applied_learning history_civics', 'music_the_arts special_needs',
       'health_sports history_civics', 'history_civics applied_learning',
       'literacy_language warmth care_hunger',
       'history_civics special_needs', 'health_sports warmth care_hunger',
       'music_the_arts health_sports',
       'applied_learning warmth care_hunger',
       'literacy_language health_sports',
       'math_science warmth care_hunger',
       'special_needs warmth care_hunger',
       'history_civics warmth care_hunger'], label="project_subject_categories"),

       gr.inputs.Dropdown(['esl literacy', 'civics_government team_sports',
       'health_wellness team_sports', 'literacy mathematics',
       'mathematics', 'literature_writing special_needs',
       'literacy special_needs', 'health_wellness',
       'literacy literature_writing', 'literacy',
       'literacy parent_involvement',
       'environmental_science health_life_science', 'special_needs',
       'applied_sciences literature_writing', 'early_development',
       'health_life_science special_needs', 'music',
       'applied_sciences mathematics', 'foreign_languages mathematics',
       'literacy other', 'health_life_science literacy',
       'economics financial_literacy', 'literature_writing',
       'team_sports', 'literature_writing mathematics',
       'health_wellness literacy', 'gym_fitness health_wellness',
       'literacy visual_arts', 'warmth care_hunger',
       'mathematics social_sciences', 'visual_arts',
       'college_career_prep literature_writing', 'environmental_science',
       'applied_sciences health_life_science', 'esl literature_writing',
       'college_career_prep', 'health_life_science mathematics',
       'character_education special_needs',
       'environmental_science mathematics', 'literacy social_sciences',
       'literature_writing visual_arts',
       'character_education health_wellness', 'music performing_arts',
       'community_service', 'gym_fitness team_sports',
       'history_geography literature_writing', 'health_life_science',
       'gym_fitness special_needs',
       'early_development literature_writing', 'performing_arts',
       'early_development mathematics', 'applied_sciences',
       'college_career_prep literacy',
       'health_life_science history_geography',
       'applied_sciences visual_arts',
       'health_wellness nutrition_education', 'literacy performing_arts',
       'mathematics special_needs', 'history_geography literacy',
       'performing_arts visual_arts', 'gym_fitness parent_involvement',
       'early_development other',
       'applied_sciences environmental_science',
       'foreign_languages literacy', 'nutrition_education',
       'literature_writing social_sciences',
       'early_development special_needs',
       'early_development nutrition_education',
       'college_career_prep special_needs', 'social_sciences visual_arts',
       'other', 'applied_sciences character_education',
       'health_wellness special_needs', 'music social_sciences', 'esl',
       'character_education community_service',
       'applied_sciences special_needs', 'gym_fitness', 'social_sciences',
       'civics_government social_sciences',
       'character_education literacy', 'esl visual_arts',
       'applied_sciences literacy', 'literature_writing other',
       'college_career_prep early_development',
       'early_development literacy', 'applied_sciences early_development',
       'character_education literature_writing',
       'character_education college_career_prep',
       'community_service health_wellness', 'health_wellness other',
       'applied_sciences college_career_prep', 'character_education',
       'college_career_prep mathematics',
       'community_service environmental_science', 'history_geography',
       'other special_needs', 'esl music', 'esl mathematics',
       'health_wellness literature_writing', 'early_development music',
       'esl special_needs', 'civics_government history_geography',
       'character_education visual_arts', 'college_career_prep other',
       'applied_sciences social_sciences',
       'history_geography mathematics', 'literacy music',
       'history_geography music', 'civics_government literacy',
       'nutrition_education other',
       'environmental_science literature_writing',
       'environmental_science visual_arts',
       'character_education mathematics',
       'environmental_science history_geography',
       'health_life_science visual_arts', 'other visual_arts',
       'music parent_involvement', 'gym_fitness nutrition_education',
       'applied_sciences parent_involvement',
       'health_wellness visual_arts',
       'environmental_science health_wellness',
       'health_life_science health_wellness',
       'applied_sciences history_geography', 'special_needs team_sports',
       'history_geography visual_arts', 'mathematics visual_arts',
       'early_development visual_arts',
       'parent_involvement special_needs', 'character_education other',
       'health_wellness mathematics',
       'early_development environmental_science',
       'health_life_science literature_writing',
       'character_education esl', 'esl early_development',
       'college_career_prep music',
       'civics_government literature_writing',
       'special_needs visual_arts', 'foreign_languages',
       'visual_arts warmth care_hunger', 'environmental_science literacy',
       'applied_sciences extracurricular', 'extracurricular other',
       'literature_writing performing_arts',
       'college_career_prep financial_literacy', 'financial_literacy',
       'character_education parent_involvement', 'applied_sciences other',
       'other parent_involvement', 'parent_involvement social_sciences',
       'mathematics music', 'parent_involvement visual_arts',
       'mathematics parent_involvement',
       'college_career_prep community_service', 'esl health_life_science',
       'environmental_science special_needs',
       'college_career_prep visual_arts',
       'financial_literacy mathematics',
       'civics_government financial_literacy',
       'health_life_science nutrition_education',
       'applied_sciences civics_government',
       'history_geography performing_arts',
       'foreign_languages literature_writing',
       'character_education early_development',
       'community_service visual_arts', 'extracurricular performing_arts',
       'character_education team_sports', 'gym_fitness performing_arts',
       'early_development performing_arts', 'health_life_science other',
       'environmental_science extracurricular',
       'college_career_prep performing_arts',
       'extracurricular visual_arts', 'music special_needs',
       'esl foreign_languages', 'health_wellness history_geography',
       'foreign_languages health_life_science',
       'extracurricular literacy', 'community_service extracurricular',
       'civics_government extracurricular',
       'early_development health_wellness',
       'community_service literature_writing',
       'community_service gym_fitness',
       'extracurricular literature_writing', 'other performing_arts',
       'civics_government community_service',
       'extracurricular special_needs', 'gym_fitness history_geography',
       'health_wellness performing_arts', 'extracurricular team_sports',
       'literacy warmth care_hunger', 'civics_government special_needs',
       'foreign_languages special_needs',
       'history_geography social_sciences', 'applied_sciences music',
       'applied_sciences esl', 'gym_fitness mathematics',
       'music visual_arts', 'economics mathematics',
       'esl environmental_science',
       'early_development parent_involvement', 'mathematics other',
       'applied_sciences performing_arts', 'extracurricular',
       'character_education extracurricular', 'health_life_science music',
       'college_career_prep parent_involvement',
       'extracurricular mathematics', 'mathematics team_sports',
       'health_life_science social_sciences',
       'history_geography special_needs', 'gym_fitness literacy',
       'environmental_science social_sciences',
       'environmental_science financial_literacy',
       'early_development team_sports',
       'civics_government health_life_science',
       'mathematics performing_arts',
       'environmental_science parent_involvement',
       'character_education gym_fitness', 'gym_fitness music',
       'economics history_geography', 'performing_arts social_sciences',
       'economics literacy', 'esl history_geography',
       'literature_writing music', 'applied_sciences team_sports',
       'civics_government environmental_science',
       'civics_government mathematics',
       'health_wellness warmth care_hunger',
       'nutrition_education special_needs',
       'extracurricular history_geography', 'economics special_needs',
       'civics_government', 'community_service social_sciences',
       'early_development health_life_science',
       'health_life_science parent_involvement',
       'applied_sciences health_wellness', 'character_education music',
       'community_service health_life_science',
       'character_education economics',
       'financial_literacy special_needs',
       'applied_sciences foreign_languages',
       'gym_fitness literature_writing', 'performing_arts team_sports',
       'literature_writing parent_involvement',
       'college_career_prep social_sciences',
       'applied_sciences community_service',
       'character_education warmth care_hunger',
       'college_career_prep nutrition_education',
       'social_sciences special_needs', 'other social_sciences',
       'extracurricular social_sciences',
       'early_development warmth care_hunger', 'economics',
       'community_service parent_involvement', 'esl financial_literacy',
       'foreign_languages health_wellness', 'health_wellness music',
       'community_service economics',
       'college_career_prep environmental_science',
       'environmental_science nutrition_education',
       'college_career_prep warmth care_hunger',
       'performing_arts special_needs', 'community_service literacy',
       'mathematics warmth care_hunger',
       'college_career_prep health_life_science',
       'early_development extracurricular',
       'extracurricular parent_involvement',
       'college_career_prep economics', 'gym_fitness visual_arts',
       'special_needs warmth care_hunger',
       'college_career_prep extracurricular',
       'literature_writing team_sports',
       'environmental_science warmth care_hunger',
       'esl parent_involvement', 'college_career_prep health_wellness',
       'college_career_prep history_geography',
       'literature_writing warmth care_hunger', 'extracurricular music',
       'gym_fitness health_life_science', 'financial_literacy other',
       'parent_involvement', 'financial_literacy literacy',
       'character_education foreign_languages', 'economics other',
       'character_education environmental_science',
       'community_service mathematics', 'history_geography other',
       'civics_government economics',
       'character_education social_sciences', 'literacy team_sports',
       'college_career_prep esl', 'applied_sciences gym_fitness',
       'civics_government visual_arts', 'applied_sciences economics',
       'environmental_science foreign_languages',
       'community_service special_needs', 'civics_government esl',
       'environmental_science music',
       'character_education civics_government',
       'economics social_sciences', 'early_development social_sciences',
       'health_wellness social_sciences', 'foreign_languages music',
       'extracurricular health_wellness',
       'college_career_prep foreign_languages',
       'early_development gym_fitness',
       'parent_involvement performing_arts', 'esl other',
       'financial_literacy history_geography',
       'environmental_science performing_arts', 'other team_sports',
       'character_education history_geography',
       'environmental_science other',
       'extracurricular health_life_science',
       'character_education nutrition_education', 'music team_sports',
       'early_development foreign_languages',
       'early_development history_geography', 'esl performing_arts',
       'esl gym_fitness', 'esl health_wellness',
       'foreign_languages performing_arts',
       'character_education performing_arts',
       'nutrition_education team_sports',
       'applied_sciences financial_literacy',
       'character_education financial_literacy',
       'health_wellness parent_involvement',
       'health_life_science warmth care_hunger',
       'foreign_languages history_geography',
       'literacy nutrition_education',
       'community_service nutrition_education',
       'financial_literacy literature_writing', 'esl social_sciences',
       'civics_government college_career_prep',
       'foreign_languages visual_arts', 'foreign_languages gym_fitness',
       'nutrition_education social_sciences',
       'history_geography parent_involvement',
       'parent_involvement warmth care_hunger',
       'character_education health_life_science', 'gym_fitness other',
       'early_development financial_literacy',
       'civics_government health_wellness',
       'applied_sciences nutrition_education',
       'college_career_prep team_sports',
       'environmental_science team_sports', 'team_sports visual_arts',
       'economics environmental_science', 'community_service other',
       'applied_sciences warmth care_hunger', 'music other',
       'community_service music', 'mathematics nutrition_education',
       'college_career_prep gym_fitness',
       'community_service performing_arts', 'economics visual_arts',
       'gym_fitness social_sciences', 'parent_involvement team_sports',
       'community_service history_geography',
       'economics literature_writing',
       'civics_government nutrition_education',
       'extracurricular gym_fitness', 'esl extracurricular',
       'financial_literacy health_wellness',
       'health_life_science team_sports',
       'financial_literacy foreign_languages',
       'nutrition_education visual_arts',
       'civics_government performing_arts', 'foreign_languages other',
       'financial_literacy social_sciences',
       'environmental_science gym_fitness',
       'economics nutrition_education',
       'community_service early_development', 'economics music',
       'foreign_languages social_sciences', 'community_service esl',
       'other warmth care_hunger', 'community_service financial_literacy',
       'financial_literacy visual_arts',
       'extracurricular foreign_languages',
       'extracurricular nutrition_education',
       'financial_literacy health_life_science',
       'financial_literacy parent_involvement',
       'extracurricular financial_literacy',
       'early_development economics', 'gym_fitness warmth care_hunger',
       'literature_writing nutrition_education',
       'history_geography team_sports',
       'health_life_science performing_arts',
       'nutrition_education warmth care_hunger',
       'economics health_life_science', 'esl nutrition_education',
       'history_geography warmth care_hunger',
       'civics_government parent_involvement', 'esl team_sports',
       'social_sciences team_sports',
       'financial_literacy performing_arts', 'esl economics',
       'civics_government foreign_languages',
       'economics foreign_languages'], label="project_subject_subcategories"),
       
    gr.inputs.Dropdown(['grades_prek_2', 'grades_6_8', 'grades_3_5', 'grades_9_12'], label="project_grade_category"),
    gr.inputs.Textbox(lines=5, label="project_title"),
    gr.inputs.Textbox(lines=5, label="project_resource_summary"),
    gr.inputs.Textbox(lines=5, label="essay"),
    gr.inputs.Slider(minimum=0, maximum=500, label="teacher_number_of_previously_posted_projects"),
    gr.inputs.Slider(minimum=0, maximum=1000, label="price"),
    gr.inputs.Slider(minimum=0, maximum=1000, label="quantity")

]

output = gr.outputs.Textbox(label="Output")

# Create the interface
interface = gr.Interface(fn=predict, inputs=inputs, outputs=output,title=" DonorChoosesPrediction Application Project")

# Launch the interface
interface.launch(share=True)
