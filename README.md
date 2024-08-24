# Extractive and Abstractive Text Summarization Project

**Description**

This project explores two primary approaches to text summarization: 

i) Extractive

ii) Abstractive. 

These methods offer complementary solutions for condensing and understanding large volumes of text, catering to diverse requirements and applications.

Extractive summarization involves selecting the most significant sentences from the original text to form a concise summary, where on the other hand, Abstractive Summarization generates new sentences that capture the essence and core ideas of the original text using advanced language models.

This README includes a brief overview, installation, Instructions, Usage, and dependencies of both Extractive and Abstractive Summarizations Seperately.



# Extractive Text Summarization

**Overview**

This Extractive Text Summarization project aims to provide a concise and informative summary of a given text document by identifying and extracting the most relevant sentences. This approach focuses on preserving the original wording and factual content of the source material, making it well-suited for applications where accuracy and authenticity are crucial.

**Features**

This Extractive Text Summarization is designed to be simple and straightforward, making it accessible for users who may not have extensive experience in natural language processing (NLP) or machine learning. Here's an explanation of the code and features used in extractive summarization: 

1. Natural Language Processing (NLP) Integration:

- The project utilizes the 'spaCy' library with the 'en_core_web_sm' model for efficient text processing, including tokenization, sentence segmentation, and stop word removal.

       nlp = spacy.load('en_core_web_sm')

       doc = nlp(rawdocs)



 2. Tokenization and Preprocessing:

- The input text is converted into tokens, focusing on words that are neither stop words nor punctuation.
- Stop words are filtered out using the 'STOP_WORDS' list from 'spaCy' to remove common, less meaningful words.

      stop_words = list(STOP_WORDS)

      tokens = [token.text.lower() for token in doc
                     if not token.is_stop and
                     not token.is_punct and
                     token.text != '\n']

- Here we have used token vaiable to convert all the text from original document to lower case as well as to prevent the stop words ans punctuation from original document text.

3. Word Frequency Calculation:

- The frequency of non-stop words is calculated using the 'Counter' class from the 'collections' module.
- The word frequencies are then normalized by dividing each word's frequency by the maximum frequency, ensuring that the scores are within a consistent range.

       # Word frequency Count
       word_freq = Counter(tokens)
       
       # Highest occuring word, Word with max. frequency
       max_freq = max(word_freq.values())

       # Normalization
       for word in word_freq.keys():
       word_freq[word] = word_freq[word]/max_freq


4. Sentence Scoring:

- The text is split into sentences using spaCy's sentence segmentation capabilities.
- Each sentence is scored based on the sum of the normalized word frequencies present in the sentence, assigning higher scores to sentences with more high-frequency words

       sent_token = [sent.text for sent in doc.sents]


       sent_score = {}
         # looping through list of sentences from sent_token.
       for sent in sent_token:
         # split each word from sentences.
       for word in sent.split():
         # checking for lowercase words from word_freq.
       if word.lower() in word_freq.keys():
                # It will check if the word exists or not in word_freq
              if sent not in sent_score.keys():
                # if it doesn't exists it will add the sentence as key and value as word from word_freq to sent_score
              sent_score[sent] = word_freq[word]
              else:
                # if it exists it increments the score of sentence with word of word_freq.
              sent_score[sent] += word_freq[word]


       # Cleaning the Sentences from sent_score by replacing '\n' to empty space ''.
       sent_scores = {}
       for key, value in sent_score.items():
       clean_key = key.replace('\n', '')
       sent_scores[clean_key] = value



5. Summary Generation:
- The number of sentences to include in the summary is determined based on a predefined proportion (30% of the total sentence count in this case).
- The top-scoring sentences are selected using the 'nlargest' function from the 'heapq' module to construct the final summary.

      num_sentences = int(len(sent_token) * 0.3)

      final_summary =  nlargest(num_sentences, sent_scores, key = sent_scores.get)



6. Output Details:
- Returns the summary as a single string of space-separated sentences.
- Provides additional information including the original document and the word count of both the original and summarized texts to assess the summarization compression

       # this diffrent sentences are joined using join function which combined make's up our text summary.
       summary = " ".join(final_summary)


       # Word count of Original Text
       len(rawdocs.split()) 

       # Word Count of Summarized text 
       len(summary.split())


- We have taken word count of both original and summarized text to showcase in our API with respective texts.

7. Data Structure Utilization:
- Constructs a DataFrame with sentences and their scores for potential analysis or debugging purposes.
- In my GitHub repo you will find python file named 'Extractive _Text_Summarization' where I have created Data Frame of Sentences and Scoring which is not mandatory in this case, It is provided for understanding purpose only.


**Benefits**

- Efficiency: The function rapidly processes text using lightweight NLP operations, making it suitable for real-time or large-scale applications.
- Simplicity: The approach uses straightforward frequency-based scoring, which is easy to understand and implement.
- Customizability: Users can adjust the summary length by modifying the proportion of sentences selected (0.3 in the code).

This implementation provides a foundational method for extractive summarization that can be extended or refined with additional NLP features or alternativescoring mechanisms for different use cases.

**Project Structure**

The project consists fo the following files:

1. `README.md` : This file contains the project overview, features, installation instructions, and usage documentation.

2. src (Source Code Directory) : 

- `Extractive_Text_Summarization.ipynb`: This Jupyter Notebook file contains the implementation of the extractive text summarization algorithm.
- `extractive_summary.py` : This Python module contains the core logic for the extractive text summarization.
- `extractive_app.py` : This Python script serves as the main application that integrates the text summarization functionality, this Python script serves as the main entry point for the web application that provides a user interface for the extractive text summarization feature.

3. web (Web Application Directory): 

- `index.html` : This HTML file serves as the main user interface for the Extractive Text Summarization project.
- `summary.html` :  This HTML file displays the summarized text generated by the text summarization algorithm.
- `style.css` : This CSS file contains the styles for the web application.

4. requirement.txt : This file lists the Python dependencies required for the project, which can be used for setting up the development environment.

5. .gitignore : This file specifies the files and directories that should be ignored by the Git version control system.


**Installation**

1. Clone the repository:

       https://github.com/KrishnaSalve/Extractive-Abstractive-Text-Summarization.git

2. Navigate to the project directory:

       cd Exctractive-Abstractive-Text-Summarization

3. Install the required dependencies:

       pip install -r requirements.txt

**Usage**

1. Run the Flask Application:
      
       python extractive_app.py

 

  This will start the web application and make it accessible at http://127.0.0.1:5000.


2. Insert Text:

- Open your web browser and navigate to http://127.0.0.1:5000.

- You will see a web interface which says, Enter your raw text.

- Enter your raw text to which you want your Summary and click on the 'Submit' button.


3. Interpret the results:

- The application will process the Raw text and display the Summarized Text.

- It will automatically take you to another user interface where you will get your Origional text as well as Summarized text with their respective Word Count.

**Result**

This section outlines the expected outcomes and effectiveness of the extractive text summarization application built using Flask and an extractive summarizer. The focus is on the application's ability to produce concise summaries that preserve the main ideas of the source text.

**Summarization Quality**

- Extractive Approach : 
  - The application employs an extractive summarization technique, which identifies and selects the most important sentences from the input text to form a summary.
  - Users can anticipate summaries that encapsulate key points and factual information directly from the original text, ensuring that significant details are retained.

- Sentence Selection : 
  - The summarizer is designed to effectively assess sentence relevance and cohesiveness, providing a summary that maintains the original context and flow as much as possible.

**User Experience**

- Web Interface : 
  - A simple and user-friendly web interface is provided, allowing users to input text and receive summaries with ease via HTML templates.
  - The interface clearly presents both the original and summarized text, along with their respective lengths, enhancing comprehension and user interaction.

**Limitations**

- Dependence on Text Structure :
  - The quality of extractive summaries is influenced by the coherence and structure of the input text. Well-structured documents lead to better summarization outcomes.
- Lack of Abstraction : 
  - As an extractive method, the summarizer does not generate novel sentences. This might leave out implicit information and nuanced insights that an abstractive summarizer could potentially capture.


Overall, this extractive summarization application provides a reliable tool for condensing information, making it suitable for quick comprehension and review of extensive or complex documents through an easy-to-use web platform.

### Abstractive Text Summarization 

**Overview**

This project demonstrates how to perform abstractive text summarization using the T5 (Text-to-Text Transfer Transformer) model from Hugging Face's Transformers library. The code utilizes the smaller variant of the T5 model (t5-small) for generating summaries of input text.

**Features**

This abstractive text summarization project using the T5 model offers several features designed to efficiently generate coherent and concise summaries from longer pieces of text. Below are the key features of this implementation:

1. Pre-trained T5 Model : 
This model Utilizes the t5-small variant from Hugging Face's Transformers library, leveraging a pre-trained model renowned for its effectiveness in text-to-text tasks.

       import torch
       from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

2. T5 Model and Tokenizer:

     **Model Initialization** :
- Below line loads the pre-trained T5 model specifically configured for conditional generation tasks using the 't5-small' variant from Hugging Face's model hub.

       model = T5ForConditionalGeneration.from_pretrained('t5-small')

- The 'T5ForConditionalGeneration' class is tailored to handle various text generation tasks, such as summarization, translation, and more.
- By specifying 't5-small', we are using a smaller, more computationally efficient version of the T5 model, which strikes a balance between performance and resource usage.


**Tokenizer Initialization** : 
- The tokenizer is responsible for converting raw text into a format the model can understand (token IDs) and vice versa (from token IDs back to text).

       tokenizer = T5Tokenizer.from_pretrained('t5-small')

- It utilizes the same 't5-small' configuration to ensure compatibility between the input text preprocessing and the model's expectations.
- The tokenizer handles tasks like tokenization, padding, and truncation when preparing text for the model.


 
**Device Specification** : 
- This line sets the computation device for PyTorch operations to the CPU (central processing unit).

       device = torch.device('cpu')

- While specifying cpu ensures that the code runs on any machine without requiring specialized hardware, if a GPU (graphics processing unit) is available, you could change this to `torch.device('cuda')` to accelerate model inference and training.

3. Text Preprocessing :

- It preprocesses input text by stripping unwanted whitespace and replacing newline characters, ensuring that the model receives clean input for optimal performance.

       preprocessed_text = text.strip().replace('\n', '')
       t5_input_text = 'summarize: ' + preprocessed_text

4. Configurable Tokenization : 
- The input text is tokenized with support for a maximum length of 512 tokens and truncation, ensuring compatibility with longer texts and enabling efficient processing.

       tokenized_text  = tokenizer.encode(t5_input_text, return_tensors = 'pt', max_length = 512, truncation = True).to(device)

- By default, the code runs on a CPU, making it accessible even on hardware without a dedicated GPU, although it can be easily adapted for GPU usage if available by changing the device configuration.


5. Flexible Summary Length : 
- Users can specify the minimum and maximum length of the generated summaries (set to 30 and 120 tokens in this example), allowing customization based on specific requirements.

       summary_ids = model.generate(tokenized_text, min_length = 30, max_length = 120)

6. Abstractive Summarization : 
- Generates novel summaries by understanding and condensing the original content, rather than merely extracting sentences, allowing for more coherent and contextually relevant summaries.

       summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)

       print(summary)

These features make the summarization tool suitable for various applications, such as summarizing articles, reports, or any extensive textual content, enhancing user productivity and information accessibility.


**Project Structure**

The project consists of following files:

1. `README.md` : Both Extractive and Abstractive text Summarization share same Readme file which consists of Project overview, features, installation instructions, and usage documentation of both extractive and abstractive separately.

2. src (Source Code Directory) : 
- `Abstractive_Text_Summarization_using_Pre-trained_Model.ipynb` : This Jupyter Notebook file contains the implementation of the abstractive text summarization algorithm.
- `abstractive_app.py` : This Python script serves as the main application that integrates the text summarization functionality, this Python script serves as the main entry point for the web application that provides a user interface for the extractive text summarization feature.

3. requirement.txt : As the extractive and abstractive text summarizations are stored in same repository they also share same requirement file, it means that, while running the command `pip install -r requirements.txt` dependecies of both the text summarizations will be install and you don't have to install dependencies twice.

4. .gitignore : By now I think you have guessed that if both the summarizations have same repo, it will also have same `.gitignore` files. Answer is 'Yes'.


**Installation**

Installation process will be same as that of Extractive Text Summarization.
In fact if you have used extractive text summarization you won't have to do this Installation process. For Installation please look out for Installation section in Extractive text summarization. As both the text summarization process exist in same repo, it will have same clone repository 'url'. During first installation, files of both the text summarizations will be cloned in your local system.

**Usage**

1. Run the Streamlit Application : 

       streamlit run abstractive_app.py

This will start the web application and make it accessible at http://localhost:5000/.

2. Insert Text : 

- Open your web browser and navigate to http://localhost:5000/.

- The application provides two main features accessible through a sidebar menu:

**Summarize Text**

- Select Summarize Text : Choose this option from the sidebar to summarize custom input text.
- Enter Text : Input your text in the provided text area.
- Submit : Click the "Submit" button to generate the summary.
- Output : 
   
   -  The left column displays the input text.
   - The right column displays the summarized version of the text.

**Summarize Document**
- Select Summarize Document : Choose this option to upload and summarize a PDF document.
- Upload PDF : Use the file uploader to upload a PDF document.
- Submit : Click the "Submit" button to extract and summarize the text from the PDF.

- Output : 
  - The left column displays the extracted text from the PDF.
  - The right column displays the summarized version of the extracted text.



**Result**

This section describes the expected outcomes and potential effectiveness of the text summarization application using the given code. The results emphasize the application's capabilities in producing coherent and concise summaries from both direct text inputs and PDF documents.

**Summarization Quality**

- Text Input Summarization : 
  - The application leverages the `txtai` summarization pipeline, which is designed to capture the essence of long text passages and distill them into shorter, meaningful summaries.
  - Users can expect summaries that provide the main points of the input text, effectively reducing information into a more digestible format while maintaining key insights.

PDF Document Summarization : 
  - For PDF documents, the application extracts text from the first page and summarizes it using the same summarization pipeline.
  - The extraction process via `pypdf` ensures that text is properly retrieved from the document, allowing for accurate summarization.
  - The summarized text will highlight the main themes and relevant information found within the uploaded document, offering a concise overview.

**User Experience**
  
- Ease of Use : 
  - The Streamlit interface provides an intuitive and interactive platform for both text and document summarization.
  - Clear output sections for both the input and summarized text enhance user understanding and visualization of the results.

**Limitations**

- Single-page PDF processing :
  - Currently, the application only processes the first page of a PDF. Future enhancements may involve expanding this capability to multiple pages.
- Text Quality Dependence :
  - The summarization quality is closely tied to the clarity and completeness of the input text. Texts with unclear or fragmented content may result in less coherent summaries.

Overall, this summarization application offers a practical and efficient tool for condensing information in textual and PDF formats, meeting various user needs for content summarization and insight extraction in an accessible web interface.




## Contributing
We welcome contributions from the community to help improve and expand the Text Summarization project. If you're interested in contributing, please follow these guidelines:

**Report Issues** : 

If you encounter any bugs, errors, or have suggestions for improvements, please open an issue on the project's GitHub repository. When reporting an issue, provide a clear and detailed description of the problem, including any relevant error messages or screenshots.

**Submit Bug Fixes or Enhancements** : 

If you've identified a bug or have an idea for an enhancement, feel free to submit a pull request. Before starting work, please check the existing issues to ensure that your proposed change hasn't already been addressed.
When submitting a pull request, make sure to:

    1. Fork the repository and create a new branch for your changes.
    2. Clearly describe the problem you're solving and the proposed solution in the pull request description.
    3. Follow the project's coding style and conventions.
    4. Include relevant tests (if applicable) to ensure the stability of your changes.
    5. Update the documentation, including the README file, if necessary.




**Improve Documentation**

If you notice any issues or have suggestions for improving the project's documentation, such as the README file, please submit a pull request with the necessary changes.

**Provide Feedback**

We value your feedback and suggestions for improving the Extractive and Abstractive Text Summarization project. Feel free to share your thoughts, ideas, or use cases by opening an issue or reaching out to the project maintainers.

**Code of Conduct**

When contributing to this project, please adhere to the project's Code of Conduct to ensure a welcoming and inclusive environment for all participants.

Thank you for your interest in contributing to the Extractive and Abstractive Text Summarization project. We appreciate your support and look forward to collaborating with you.


### Contact

If you have any questions, feedback, or would like to get in touch with the project maintainers, you can reach us through the following channels:

- **Project Maintainer**

Name : Krishna Salve 

Email : krishnasalve97@gmail.com

Linkedin : Krishna Salve

GitHub : KrishnaSalve



- **Project Repository**

     https://github.com/KrishnaSalve/Extractive-Abstractive-Text-Summarization




