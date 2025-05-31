import streamlit as st
from generation import QuestionsGenerator

# Заголовок интерфейса
st.set_page_config(page_title="AIMovieMind - Генератор заданий")
st.title("🎬 AIMovieMind: Генерация заданий по транскриптам")

# Ввод текста
user_input = st.text_area("Вставьте транскрипт фильма, подкаста или сериала:", height=300)

# Выбор количества вопросов
num_questions = st.slider("Сколько вопросов сгенерировать?", min_value=3, max_value=10, value=5)

# Кнопка запуска
if st.button("🔍 Сгенерировать вопросы"):
    if not user_input.strip():
        st.warning("Пожалуйста, вставьте текст для анализа.")
    else:
        with st.spinner("Генерация вопросов, подождите..."):
            try:
                generator = QuestionsGenerator(init_llms=['openai'])
                result = generator.generate(user_input, num_questions, llm='openai')

                if isinstance(result, list):
                    for i, q in enumerate(result, 1):
                        st.markdown(f"**Вопрос {i}:**")
                        st.write(q)
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"Ошибка генерации: {e}")

st.markdown("---")
st.caption("Создано с ❤️для людей")
