# 🎯 AURA Diagnostics - Hackathon Demo Guide

## 🚀 Quick Start (2 minutes)

### 1. Start Both Servers
```bash
# Terminal 1: Backend
cd /Users/suditipradhan/Desktop/imaging
python main.py

# Terminal 2: Frontend  
cd /Users/suditipradhan/Desktop/imaging/frontend
npm run dev
```

### 2. Open Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 🎬 Demo Script (5 minutes)

### Opening (30 seconds)
> "AURA Diagnostics is a complete medical imaging analysis system that combines AI-powered vision models with an intuitive web interface. Let me show you how it works."

### Demo Flow

#### 1. Upload Demo (1 minute)
- **Action**: Drag and drop a medical image (X-ray, skin photo, etc.)
- **Say**: "Users can simply drag and drop medical images - X-rays, MRIs, skin conditions, anything."
- **Show**: Image preview appears with remove option

#### 2. Question Input (30 seconds)
- **Action**: Type a medical question like "Do you see pneumonia?" or "Could this be eczema?"
- **Say**: "They can ask specific medical questions to get targeted analysis."

#### 3. AI Analysis (1 minute)
- **Action**: Click "Analyze" button
- **Say**: "Our system uses Replicate's LLaVA-Med model with Hugging Face BiomedCLIP as fallback for reliability."
- **Show**: Loading spinner, then structured results

#### 4. Results Display (1 minute)
- **Action**: Point to each section of results
- **Say**: "Results are displayed in a chat-style interface with:"
  - **Findings**: Bullet-pointed medical observations
  - **Confidence**: Visual percentage bar
  - **Recommendations**: Actionable medical advice
  - **Question Interpretation**: Shows AI understood the question

#### 5. Camera Demo (1 minute)
- **Action**: Click "Use Camera" button
- **Say**: "For real-time scenarios, users can capture images directly."
- **Show**: Live camera feed, capture photo, analyze

#### 6. Error Handling (30 seconds)
- **Action**: Show what happens with invalid files or API failures
- **Say**: "The system gracefully handles errors with user-friendly messages."

## 🎯 Key Talking Points

### Technical Highlights
- **Full-Stack**: Next.js frontend + FastAPI backend
- **AI Integration**: Dual VLM approach for reliability
- **Real-Time**: Live camera capture with WebRTC
- **Responsive**: Works on all devices
- **Production-Ready**: Comprehensive error handling

### Business Value
- **Accessibility**: Makes medical AI available to everyone
- **Speed**: Instant analysis vs. waiting for specialists
- **Accuracy**: Dual AI models for better reliability
- **Scalability**: Cloud-ready architecture

### Demo Impact
- **Visual**: Beautiful, professional medical interface
- **Interactive**: Live camera and drag-drop functionality
- **Practical**: Real medical use cases
- **Impressive**: AI analysis with confidence scores

## 🛠️ Troubleshooting

### If Backend Fails
```bash
pip install fastapi uvicorn python-multipart replicate pillow requests aiohttp
python main.py
```

### If Frontend Fails
```bash
cd frontend
npm install
npm run dev
```

### If Camera Doesn't Work
- Use HTTPS or localhost (required for camera access)
- Check browser permissions
- Use file upload as backup

## 📱 Demo Tips

### Before Demo
- [ ] Test both upload and camera functionality
- [ ] Prepare 2-3 sample medical images
- [ ] Check internet connection (for AI APIs)
- [ ] Have backup screenshots ready

### During Demo
- [ ] Start with most impressive feature (AI analysis)
- [ ] Show mobile responsiveness
- [ ] Highlight error handling
- [ ] Mention scalability and integration potential

### Sample Questions to Ask
- "Do you see pneumonia in this chest X-ray?"
- "Could this skin condition be eczema?"
- "What abnormalities do you detect?"
- "Is this fracture visible in the bone?"

## 🏆 Closing Statement

> "AURA Diagnostics demonstrates how modern web technologies can make advanced medical AI accessible to healthcare providers worldwide. The modular architecture makes it perfect for integration into existing hospital systems, and the dual AI approach ensures reliability in critical medical scenarios."

---

**Total Demo Time**: 5-7 minutes
**Setup Time**: 2 minutes
**Wow Factor**: High 🚀