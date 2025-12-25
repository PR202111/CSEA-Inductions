🚩 AI Flag Detector & Meme Sentiment Translator

This repo contains two AI projects built for CSEA Inductions, focusing on end-to-end ML + LLM systems with clean APIs and fast demos.

1️⃣ AI Green / Red Flag Detector (Dating Bios)

Analyzes dating app bios to identify green, yellow, and red flags, explain the reasoning, and rewrite the bio into a healthier version using an LLM.

Features

DistilBERT-based multi-label classifier

Scores: healthy, neutral, toxic, narcissistic, cringe

Keyword/flag extraction

LLM-powered bio rewrite (wholesome / confident / funny)

APIs

POST /classify → flags, scores, meta
POST /rewrite  → improved_bio, alternatives

2️⃣ Meme Sentiment Translator

Uploads a meme image, understands its text and vibe, and rewrites it in different comedic styles.

Features

OCR using PaddleOCR

Meme template detection via CLIP

Sentiment & vibe classification

LLM rewrites: sarcastic, wholesome, Gen-Z

APIs

POST /meme-read
POST /meme-emotion
POST /meme-rewrite



Author

Built by Prashant for CSEA Inductions