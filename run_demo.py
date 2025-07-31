from steme_core import STEME, STEMEnx

texts = ["I love music", "Sound makes me happy", "Silence is golden"]
query = "I enjoy listening to tunes"

print("Top matches:")
for sim, match in STEME(query, texts):
    print(f"{sim:.3f} → {match['content']}")

print("\nSemantic cohesion of list:")
print(STEMEnx(texts))
