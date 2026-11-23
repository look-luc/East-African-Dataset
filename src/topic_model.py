from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    """
    From my observations, models tend to translate Luganda proverbs into English by focusing mainly on the direct or literal meaning of the words, preserving the original expressions as closely as possible without necessarily capturing the deeper or figurative meaning intended in the proverb  leads to the loss or distortion of several aspects of the original Luganda meaning.
    The six aspects I identified during my review are:
    Figurative Meaning: Models translate the words literally but fail to capture the deeper message or lesson conveyed by the proverb.
    Cultural Context:** Models lose cultural references, traditions, beliefs and social experiences embedded in the original Luganda proverb.
    Symbolism: Models translate animals, objects and everyday activities literally without recognising what they represent or symbolise in Ganda culture.
    Implied Meaning: Models fail to capture underlying advice, warnings, criticism or moral lessons that are not explicitly stated.
    Ambiguity: Models settle on one interpretation of a proverb while overlooking other possible meanings or interpretations.
    Tone and Intention: Models fail to preserve humour, irony, sarcasm and indirect criticism, sometimes changing the speaker's original intention.
    """,
    """
    Oh this I need to go back and think a little.
    Sometimes models get lost on literal meaning, cultural understanding..word play, correlation to events and exact meanings. Gikuyu is very convoluted, phrases in general they don't mean exactly what they say.
    A model can even miss out on a twist on a word that makes sense in the real world spoken version.
    I would need to go back and check all phrases but in general. Models miss on cultural grounding, local context,word play,and societal context. They are all related.
    For instance search
    Kana ka nikora kona kora kora nako kana ka nikora kona kora
    Although this is a tongue twister no model can get it right yet.
    """
]

def LDA(documents:list[str]=documents, num_topics:int=3):
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words="english")
    dtm = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    # 3. Display the top words for each topic
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
      top_words = [words[i] for i in topic.argsort()[:-6:-1]]
      print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
