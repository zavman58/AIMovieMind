<p align="center"><img src=img/logo.png width=450px></p>

<p align="center"><b><b>AIMovieMind</b></b> - сайт для генерации вопросов по транскрипции фильма/сериала/подкаста. Оно помогает пользователям лучше понимать и анализировать информацию, предлагая вопросы для проверки владения материалом после его изучения.</p>

<div align="center">
  <h3>🧠 Core ML & NLP</h3>
  <p>
    <img src="https://img.shields.io/badge/PyTorch-2.6.0+cu118-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Transformers-4.50.3-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers">
    <img src="https://img.shields.io/badge/Sentence%20Transformers-4.0.1-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="SentenceTransformers">
  </p>
  
  <h3>📊 Data & Math</h3>
  <p>
    <img src="https://img.shields.io/badge/Numpy-2.2.4-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/SciPy-1.15.2-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
    <img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  </p>
  
  <h3>🌐 Web & API</h3>
  <p>
    <img src="https://img.shields.io/badge/Streamlit-1.45.0-000000?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/OpenAI-1.70.0-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  </p>
  
  <h3>🛠 Utilities</h3>
  <p>
    <img src="https://img.shields.io/badge/tqdm-4.67.1-FFC107?style=for-the-badge&logo=python&logoColor=black" alt="tqdm">
    <img src="https://img.shields.io/badge/HuggingFace%20Hub-0.30.1-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace Hub">
  </p>
</div>

## Как работает алгоритм?

1. Текст по абзацам разбивается на ```чанки```. Чанки фильтруются и векторизуются.

<p align="center"><img src=img/chunk_split.png width=400px></p>

2. Эмбеддинги чанков кластеризуется на ```количество_чанков * 0.01``` кластеров. Из каждого кластера берётся ближайший к центру кластера объект.
<p align="center"><img src=img/clustering.png width=400px></p>

3. Центральные объекты каждого кластера передаются на вход ```LLM```.

## Обучение модели


В рамках проекта была дообучена языковая модель [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) на генерацию вопросов по фрагментам текста

## Возможности
- Обработка длинных текстов
- Генерация вопросов на основе содержимого
- Поиск устойчивых выражений, фразовых глаголов, жаргонов

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone [https://github.com/zavman58/AIMovieMind.git]
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование
Перейдите в папку ```src```
```bash
cd src
```
Создайте файл ```.env``` и добавьте в него свой ключ DeepSeek API

```
DEEPSEEK_API_KEY=<Your API key>
```

### Python интерфейс

```python
from generation import QuestionsGenerator

with open('text.txt', 'r', encoding='utf8') as f:
    text = f.read()

# Генерация 10 вопросов по тексту
generator = QuestionsGenerator()
questions = generator.generate(text, 10)
```

```QuestionsGenerator.generate(text: str, questions_num: int) -> list[dict]``` возвращает список со словарями. Каждый словарь представляет из себя описание вопроса. Пример словаря:

```
{
'question': 'Кто является автором романа «Война и мир»?',
'answers': [
            {
              'answer': 'Александр Пушкин',
              'is_correct': false
            },
            {
              'answer': 'Лев Толстой',
              'is_correct': true
            },
            {
              'answer': 'Фёдор Достоевский',
              'is_correct': false
            },
            {
              'answer': 'Николай Гоголь',
              'is_correct': false
            }
          ]
}
```

### Графический Web интерфейс
1. Запустите сервер
   ```bash
   python app.py
   ```
2. Перейдите на запущенный локальный сервер (пример ```http://127.0.0.1:8080/```)


