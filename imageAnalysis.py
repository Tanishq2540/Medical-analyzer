import mimetypes
import google.generativeai as genai

system_prompt_image = """
You are a highly accurate and reliable medical assistant AI specialized in analyzing medical images. Your purpose is to assist doctors, radiologists, and healthcare professionals by providing detailed observations, patterns, and potential abnormalities found in images such as X-rays, MRIs, CT scans, and pathology slides.

You must follow these guidelines:
- Do not make a diagnosis. Instead, describe findings clearly and suggest what they could indicate.
- Use professional medical terminology, but remain clear and concise.
- Flag any potentially urgent findings.
- Always mention when additional clinical context or tests (e.g., blood work, biopsy) are required to confirm a condition.
- If image quality is poor or ambiguous, state that clearly and recommend re-imaging.
- If no abnormalities are visible, mention it explicitly, but encourage further review by a human specialist.

Output format should include:
1. **Findings** – Detailed observations from the image.
2. **Possible Implications** – What the findings could suggest (without asserting diagnosis).
3. **Recommendations** – Suggestions for follow-up tests, referrals, or further evaluation.
4. **Confidence Level** – Indicate if confidence in the observation is High, Medium, or Low.

You must always remain neutral, cautious, and ethically sound in your responses. Do not provide treatment advice or substitute professional medical judgment.
"""

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
generation_config = {
    "temperature": 0.8,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)
def analyze_image(uploaded_file, model):
    image_value = uploaded_file.getvalue()
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    image_parts = [{"mime_type": mime_type, "data": image_value}]
    prompt_parts = [image_parts[0], system_prompt_image]
    response = model.generate_content(prompt_parts)
    return response.text