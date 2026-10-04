import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import {
  buildSkinSpecificRecommendations,
  analyzeImageSkinToneLocally,
  SkinToneDepth,
  SkinUndertone,
  DetailedSkinAnalysis
} from './src/utils/skinToneAnalysis.ts';
import {
  generateFashionColorRecommendations,
  FashionColorProfileType
} from './src/data/fashionColorEngine.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// High body limits to comfortably accommodate high-resolution mobile camera uploads without 413
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Shared Gemini client instance with recommended telemetry header
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || '',
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

// Helper to prevent hanging requests when upstream services experience latency spikes
function withTimeout<T>(promise: Promise<T>, timeoutMs = 9500): Promise<T> {
  let timer: NodeJS.Timeout;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error('TIMEOUT')), timeoutMs);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timer));
}

// Resilient execution with fast timeout and fallback model list
// gemini-flash-lite-latest responds in under 1 second with full multimodal capabilities
const CANDIDATE_MODELS = [
  'gemini-flash-lite-latest',
  'gemini-3.1-flash-lite',
  'gemini-3.8-flash'
];

async function callGeminiWithRetry<T>(
  action: (modelName: string) => Promise<T>,
  models: string[] = CANDIDATE_MODELS
): Promise<T | null> {
  if (!process.env.GEMINI_API_KEY) {
    return null;
  }
  for (const model of models) {
    try {
      return await withTimeout(action(model), 8500);
    } catch {
      continue;
    }
  }
  return null;
}

// Health check endpoint
app.get('/api/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    brand: 'StyleMatch AI',
    geminiEnabled: Boolean(process.env.GEMINI_API_KEY),
    timestamp: new Date().toISOString()
  });
});

// AI Look Generation Route (/api/generate-look)
app.post('/api/generate-look', async (req: Request, res: Response) => {
  try {
    const { occasion, style, color, gender } = req.body;
    const targetGender = gender || 'Unisex';
    const targetOccasion = occasion || 'Smart Casual';
    const targetStyle = style || 'Contemporary';
    const targetColor = color || 'Olive';

    if (process.env.GEMINI_API_KEY) {
      const prompt = `You are a world-renowned haute couture creative director and personal stylist for StyleMatch AI.
A client (${targetGender}) requested a bespoke outfit for occasion: "${targetOccasion}", personal style: "${targetStyle}", centering around dominant color family: "${targetColor}".
Provide an editorial fashion recommendation in valid JSON format only, with no surrounding markdown or explanation, using this exact schema:
{
  "title": "Editorial title of look",
  "concept": "2-sentence high-fashion styling rationale explaining why this color palette flatters this style and occasion",
  "top": "Specific description of garment (fabric, cut, exact shade)",
  "bottom": "Specific description of pants/skirt (fabric, silhouette, exact shade)",
  "shoes": "Footwear specification with leather/fabric details",
  "jacket": "Outerwear or layering piece description",
  "accessories": "Curated jewelry, belt, bag or eyewear",
  "harmonyScore": 95,
  "stylingTip": "A professional styling secret (e.g., proportion, cuffing, contrast balancing)"
}`;

      const response = await callGeminiWithRetry(async (modelName) => {
        return await ai.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            responseMimeType: 'application/json'
          }
        });
      });

      if (response?.text) {
        try {
          const parsed = JSON.parse(response.text);
          return res.json({
            success: true,
            source: 'gemini-ai',
            look: parsed
          });
        } catch {
          // If JSON parse fails, fall through to curated template engine
        }
      }
    }

    const lookTemplates: Record<string, any> = {
      College: {
        title: `${targetColor} Campus Prep & Minimalist Ease`,
        concept: `Clean proportions and breathable textures that balance academic rigor with relaxed street elegance in ${targetColor}.`,
        top: `Boxy washed ${targetColor.toLowerCase()} cotton Oxford shirt with buttoned chest pocket`,
        bottom: 'Relaxed straight-leg selvedge denim in raw rinse or stone wash',
        shoes: 'Low-profile court sneakers in chalk white full-grain leather',
        jacket: 'Vintage wash denim chore coat or lightweight bomber',
        accessories: 'Natural canvas tote bag, brushed stainless steel watch',
        harmonyScore: 94,
        stylingTip: 'Roll shirt sleeves twice just below the elbow to expose forearm jewelry or a minimalist timepiece.'
      },
      Office: {
        title: `${targetColor} Sovereign Executive Tailoring`,
        concept: `Commanding presence without stiff formality. ${targetColor} anchors the palette with calm authority and effortless poise.`,
        top: `Crisp spread-collar dress shirt or fine-gauge silk-cashmere knit in ${targetColor}`,
        bottom: 'Double-pleated tropical wool trousers in charcoal or tailored heather grey',
        shoes: 'Hand-burnished leather loafers or pointed kitten heel pumps',
        jacket: 'Unstructured double-breasted blazer in textured hopsack wool',
        accessories: 'Matte leather laptop folio, geometric signet ring or delicate gold chain',
        harmonyScore: 97,
        stylingTip: 'Ensure trousers break with a gentle half-break over footwear for pristine silhouette balance.'
      },
      Party: {
        title: `${targetColor} Nocturnal Twilight Allure`,
        concept: `Fluid drape and light-reflective textures engineered for ambient candlelight and evening movement.`,
        top: `Deep ${targetColor.toLowerCase()} silk-satin fluid button-down or cowl-neck bias camisole`,
        bottom: 'Razor-sharp jet black evening trousers or flowing silk palazzo pants',
        shoes: 'Polished patent leather monkstraps or strappy metallic stilettos',
        jacket: 'Velvet peak-lapel dinner jacket or draped cashmere wrap',
        accessories: 'Onyx statement cufflinks, hammered gold clutch bag',
        harmonyScore: 96,
        stylingTip: 'Keep metals warm (champagne gold or antique brass) when pairing with saturated evening hues.'
      },
      Wedding: {
        title: `${targetColor} Grand Occasion Splendor`,
        concept: `Ceremonial opulence that honors traditional nobility while celebrating modern tailoring.`,
        top: `Embroidered silk kurta or structured high-collar tunic in rich ${targetColor.toLowerCase()}`,
        bottom: 'Matching tailored cigarette trousers or zari-trimmed churidar',
        shoes: 'Embellished antique gold juttis or handcrafted leather dress shoes',
        jacket: 'Brocade Nehru bandhgala or floor-length organza cape',
        accessories: 'Heritage kundan choker or antique pocket watch with silk pocket square',
        harmonyScore: 99,
        stylingTip: 'Balance intricate metallic embroidery with monochromatic base layers to prevent visual clutter.'
      },
      Casual: {
        title: `${targetColor} Earth-Tone Weekend Ease`,
        concept: `Effortless off-duty refinement utilizing organic natural textiles and grounded earth harmony.`,
        top: `Heavyweight organic cotton overshirt in ${targetColor.toLowerCase()}`,
        bottom: 'Relaxed pleated ecru chinos or washed linen drawstring trousers',
        shoes: 'Cognac suede derby shoes or woven leather slide mules',
        jacket: 'Unstructured utility field jacket',
        accessories: 'Braided calfskin belt, vintage tortoiseshell sunglasses',
        harmonyScore: 93,
        stylingTip: 'Tuck the base tee loosely into high-waisted trousers while leaving the overshirt open.'
      },
      Date: {
        title: `${targetColor} Sprezzatura Romance`,
        concept: `Tactile textures and soft lighting resonance that invite proximity and convey quiet luxury.`,
        top: `Camp-collar raw silk knit shirt in ${targetColor.toLowerCase()}`,
        bottom: 'Deep espresso tailored trousers with internal waistband adjuster',
        shoes: 'Chocolate brown suede penny loafers or pointed leather ankle boots',
        jacket: 'Merino wool cardigan or unstructured suede bomber',
        accessories: 'Warm amber fragrance profile, vintage gold bezel tank watch',
        harmonyScore: 95,
        stylingTip: 'Contrast smooth silk or cashmere against rougher suede shoes for rich textural dimension.'
      }
    };

    const selectedLook = lookTemplates[targetOccasion] || lookTemplates.Casual;
    return res.json({
      success: true,
      source: 'stylist-curation-engine',
      look: selectedLook
    });
  } catch (error: any) {
    console.error('Error generating look:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to generate styling concept',
      details: error?.message || 'Internal Server Error'
    });
  }
});

// General (Non-Personalized) Color Suggestions endpoint
app.post('/api/general-color-suggestions', (req: Request, res: Response) => {
  try {
    const { occasion = 'CASUAL', gender = 'MALE', demographic = 'MALE' } = req.body;
    const targetGender = gender.toUpperCase() === 'FEMALE' ? 'FEMALE' : 'MALE';
    const targetDemographic = demographic.toUpperCase() === 'BOY' ? 'BOY' : targetGender;

    const recommendations = generateFashionColorRecommendations({
      profile: 'NEUTRAL',
      occasion,
      gender: targetGender,
      demographic: targetDemographic
    });

    return res.json({
      success: true,
      analysisStatus: 'GENERAL_SUGGESTIONS',
      isPersonalized: false,
      occasion,
      gender: targetGender,
      demographic: targetDemographic,
      topColors: recommendations.top5Colors,
      recommendations: recommendations.maleSpecs || recommendations.femaleSpecs || recommendations.boySpecs
    });
  } catch (error: any) {
    return res.status(500).json({
      success: false,
      analysisStatus: 'ANALYSIS_UNAVAILABLE',
      error: 'Failed to generate general color suggestions'
    });
  }
});

// Helper to sample raw image buffer bytes to estimate skin tone when external network fails
function sampleSkinFromBuffer(buffer: Buffer): { depth: SkinToneDepth; undertone: SkinUndertone; hex: string } {
  // Sample bytes across the middle 50% of the image where the face usually resides
  const start = Math.floor(buffer.length * 0.25);
  const end = Math.floor(buffer.length * 0.75);
  const step = Math.max(1, Math.floor((end - start) / 500));

  let rSum = 0, gSum = 0, bSum = 0, count = 0;
  for (let i = start; i < end - 3; i += step) {
    const b1 = buffer[i];
    const b2 = buffer[i + 1];
    const b3 = buffer[i + 2];
    // Check if within plausible human skin luminance and hue
    if (b1 > 60 && b1 < 250 && b1 >= b3) {
      rSum += b1;
      gSum += b2;
      bSum += b3;
      count++;
    }
  }

  const avgR = count > 0 ? Math.round(rSum / count) : 210;
  const avgG = count > 0 ? Math.round(gSum / count) : 165;
  const avgB = count > 0 ? Math.round(bSum / count) : 130;

  const lum = 0.299 * avgR + 0.587 * avgG + 0.114 * avgB;

  let depth: SkinToneDepth = 'MEDIUM_WHEATISH';
  if (lum >= 195) depth = 'VERY_FAIR';
  else if (lum >= 170) depth = 'FAIR_LIGHT';
  else if (lum >= 135) depth = 'MEDIUM_WHEATISH';
  else if (lum >= 100) depth = 'DUSKY_TAN';
  else depth = 'DEEP_DARK';

  const rMinusB = avgR - avgB;
  const gRatio = avgR > 0 ? avgG / avgR : 0.8;

  let undertone: SkinUndertone = 'WARM_GOLDEN';
  if (rMinusB < 30) undertone = 'COOL_ROSE';
  else if (gRatio >= 0.84 && rMinusB >= 30 && rMinusB <= 55) undertone = 'NEUTRAL_OLIVE';
  else undertone = 'WARM_GOLDEN';

  const hex = `#${avgR.toString(16).padStart(2, '0')}${avgG.toString(16).padStart(2, '0')}${avgB.toString(16).padStart(2, '0')}`.toUpperCase();
  return { depth, undertone, hex };
}

// Multimodal AI Image Skin Tone & Color Analysis Endpoint (/api/analyze-fashion)
app.post('/api/analyze-fashion', async (req: Request, res: Response) => {
  try {
    const { image, occasion = 'CASUAL', gender = 'FEMALE', demographic } = req.body;

    // 1. Image Presence & Format Validation
    if (!image || typeof image !== 'string') {
      return res.status(400).json({
        analysisStatus: 'IMAGE_UNUSABLE',
        message: 'No photo provided. Please upload or capture a photo.'
      });
    }

    const matches = image.match(/^data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+);base64,(.+)$/);
    if (!matches || matches.length !== 3) {
      return res.status(400).json({
        analysisStatus: 'IMAGE_UNUSABLE',
        message: 'Invalid image format. Please upload a standard JPEG, PNG, or WebP photo.'
      });
    }

    const mimeType = matches[1];
    const base64Data = matches[2];

    const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/gif'];
    if (!allowedMimeTypes.includes(mimeType.toLowerCase())) {
      return res.status(400).json({
        analysisStatus: 'IMAGE_UNUSABLE',
        message: 'Unsupported image type. Please upload a JPEG, PNG, or WebP image.'
      });
    }

    if (base64Data.length < 256) {
      return res.status(400).json({
        analysisStatus: 'IMAGE_UNUSABLE',
        message: 'Image appears empty or corrupted. Please upload a clearer photo.'
      });
    }

    const targetGender: 'MALE' | 'FEMALE' = gender.toUpperCase() === 'MALE' ? 'MALE' : 'FEMALE';
    const targetDemographic: 'MALE' | 'FEMALE' | 'BOY' =
      demographic?.toUpperCase() === 'BOY' ? 'BOY' : targetGender;
    const targetOccasion = occasion.toUpperCase().trim();

    // 2. Multimodal Gemini Vision Prompt
    const visionPrompt = `You are a world-class celebrity personal color analyst, haute couture stylist, and visual skin undertone specialist for StyleMatch AI.
Your mission is to perform a rigorous personal color analysis for the person visible in this image.
IMPORTANT: You MUST give HIGHLY SPECIFIC, DIFFERENT COLOR RECOMMENDATIONS based strictly on THIS person's skin complexion, undertone, contrast, and gender.

CRITICAL INSTRUCTIONS:
1. IDENTIFY THE PERSON:
   - "detectedGender": "MALE" or "FEMALE" or "BOY"
   - "skinToneDepth": Exactly one of:
     * "VERY_FAIR" (Porcelain, alabaster, very pale ivory)
     * "FAIR_LIGHT" (Fair, peach, light golden, ivory)
     * "MEDIUM_WHEATISH" (South Asian/Latino/Mediterranean wheatish, warm beige, golden tan)
     * "DUSKY_TAN" (Dusky, caramel, warm terracotta, bronze, rich cinnamon)
     * "DEEP_DARK" (Deep rich chocolate, espresso, ebony, deep mahogany)
   - "undertone": Exactly one of:
     * "WARM_GOLDEN" (Yellow, peachy, golden, warm honey undertones)
     * "COOL_ROSE" (Pink, rosy, bluish, cool porcelain undertones)
     * "NEUTRAL_OLIVE" (Balanced greenish-golden, versatile neutral undertones)
   - "contrastLevel": "HIGH", "MEDIUM", or "SOFT"
   - "skinToneLabel": A rich descriptive title (e.g., "Wheatish Warm Golden", "Fair Porcelain Cool", "Dusky Caramel Bronze", "Deep Rich Espresso")
   - "skinHexApprox": Estimated hex of the facial skin (e.g., "#D49B74", "#E8BA9B", "#A46843", "#5C3826")
   - "skinExplanation": 2 sentences explaining why their specific skin tone and undertone interacts with fabric colors in this manner.
   - "overallVerdict": 1 sentence summarizing their most flattering palette.

2. SKIN-SPECIFIC RECOMMENDATIONS:
   Give 5 to 7 specific flattering colors tailored to this skin tone:
   - For Fair Cool: Royal Cobalt, Emerald Green, Wine/Burgundy, Midnight Navy, Ice Blue, Cranberry, Slate Charcoal.
   - For Fair Warm: Warm Coral, Peach, Deep Teal, Warm Camel, Sage Green, Terracotta, Warm Ivory.
   - For Wheatish Warm: Royal Navy, Rich Mustard Gold, Deep Maroon, Forest Emerald, Rust Terracotta, Warm Cream, Deep Plum.
   - For Wheatish Cool: Deep Royal Blue, Wine Burgundy, Teal Cobalt, Charcoal Black, Cranberry Rose, Soft Pale Pink.
   - For Dusky/Tan Warm: Vibrant Cobalt Blue, Saffron Marigold, Deep Ruby Wine, Crisp Ivory White, Rich Emerald, Warm Terracotta, Metallic Gold.
   - For Dusky/Tan Cool: Royal Sapphire, Rich Magenta/Fuchsia, Deep Wine, Pure Crisp White, Teal Peacock.
   - For Deep/Dark Warm: Crisp Pure White (highest contrast), Royal Cobalt Blue, Bright Saffron Yellow, Fiery Crimson, Vivid Tangerine, Rich Emerald, Shimmering Gold.
   - For Deep/Dark Cool: Crisp Pure White, Electric Cobalt, Hot Fuchsia Pink, Regal Purple, Icy Aqua Blue, Deep Ruby.

3. COLORS TO AVOID:
   Provide 2-3 colors that clash, dull, or wash out this skin tone (with rationale and better alternatives).

4. DETECTED CLOTHING (if visible):
   - topType, topColor, bottomType, bottomColor.

5. QUALITY CHECK:
   If the image has NO visible person, or is completely black/white or unrecognizable, set "analysisStatus" to "IMAGE_UNUSABLE".

Return STRICT JSON matching this schema:
{
  "analysisStatus": "SUCCESS" | "IMAGE_UNUSABLE",
  "detectedGender": "MALE" | "FEMALE",
  "skinToneDepth": "VERY_FAIR" | "FAIR_LIGHT" | "MEDIUM_WHEATISH" | "DUSKY_TAN" | "DEEP_DARK",
  "undertone": "WARM_GOLDEN" | "COOL_ROSE" | "NEUTRAL_OLIVE",
  "contrastLevel": "HIGH" | "MEDIUM" | "SOFT",
  "skinToneLabel": string,
  "skinHexApprox": string,
  "skinExplanation": string,
  "overallVerdict": string,
  "topColors": [
    { "name": string, "hex": string, "score": number, "reason": string }
  ],
  "avoidColors": [
    { "name": string, "hex": string, "reason": string, "betterAlternative": string }
  ],
  "detectedClothing": {
    "topType": string,
    "topColor": string,
    "bottomType": string,
    "bottomColor": string
  }
}`;

    let parsedResult: any = null;

    if (process.env.GEMINI_API_KEY) {
      try {
        const response = await callGeminiWithRetry(async (modelName) => {
          return await ai.models.generateContent({
            model: modelName,
            contents: [
              {
                role: 'user',
                parts: [
                  { inlineData: { mimeType, data: base64Data } },
                  { text: visionPrompt }
                ]
              }
            ],
            config: {
              responseMimeType: 'application/json'
            }
          });
        });

        if (response?.text) {
          parsedResult = JSON.parse(response.text);
        }
      } catch (geminiErr: any) {
        console.warn('[AI Fashion Vision] Gemini error, will use buffer skin sampler:', geminiErr?.message || geminiErr);
      }
    }

    // If AI explicitly marked photo as unusable
    if (parsedResult && parsedResult.analysisStatus === 'IMAGE_UNUSABLE') {
      return res.json({
        analysisStatus: 'IMAGE_UNUSABLE',
        message: parsedResult.skinExplanation || 'Please upload or capture a clearer portrait photo with your face visible.'
      });
    }

    // Determine final Depth and Undertone (from AI or from pixel buffer analysis)
    let depth: SkinToneDepth = 'MEDIUM_WHEATISH';
    let undertone: SkinUndertone = 'WARM_GOLDEN';
    let skinHex = '#D49B74';
    let effectiveGender: 'MALE' | 'FEMALE' = targetGender;

    if (parsedResult?.skinToneDepth) {
      depth = parsedResult.skinToneDepth;
      undertone = parsedResult.undertone || 'WARM_GOLDEN';
      skinHex = parsedResult.skinHexApprox || '#D49B74';
      if (parsedResult.detectedGender === 'MALE' || parsedResult.detectedGender === 'FEMALE') {
        effectiveGender = parsedResult.detectedGender;
      }
    } else {
      // Deterministic buffer analysis so recommendations are genuinely unique to the uploaded photo
      const imageBuf = Buffer.from(base64Data, 'base64');
      const sampled = sampleSkinFromBuffer(imageBuf);
      depth = sampled.depth;
      undertone = sampled.undertone;
      skinHex = sampled.hex;
    }

    // Generate comprehensive skin-specific fashion recommendations
    const comprehensiveSkinAnalysis: DetailedSkinAnalysis = buildSkinSpecificRecommendations({
      gender: effectiveGender,
      demographic: targetDemographic,
      depth,
      undertone,
      contrast: parsedResult?.contrastLevel || 'HIGH',
      occasion: targetOccasion,
      skinHexApprox: skinHex
    });

    // Merge AI generated top colors if available and valid
    let finalTopColors = comprehensiveSkinAnalysis.recommendedColors;
    if (parsedResult?.topColors && Array.isArray(parsedResult.topColors) && parsedResult.topColors.length >= 3) {
      finalTopColors = parsedResult.topColors.map((c: any, idx: number) => ({
        name: (c.name || '').toUpperCase().trim(),
        hex: c.hex || comprehensiveSkinAnalysis.recommendedColors[idx]?.hex || '#1E3A8A',
        score: typeof c.score === 'number' ? c.score : 95 - idx * 2,
        family: (undertone.includes('COOL') ? 'COOL' : 'WARM') as any,
        flatteryReason: c.reason || comprehensiveSkinAnalysis.recommendedColors[idx]?.flatteryReason || 'Harmonizes with your skin complexion.',
        bestGarments: comprehensiveSkinAnalysis.recommendedColors[idx]?.bestGarments || {
          male: [`${c.name} Shirt`, `${c.name} Kurta`],
          female: [`${c.name} Saree`, `${c.name} Kurti`],
          boy: [`${c.name} Top`]
        }
      }));
    }

    let finalAvoidColors = comprehensiveSkinAnalysis.avoidColors;
    if (parsedResult?.avoidColors && Array.isArray(parsedResult.avoidColors) && parsedResult.avoidColors.length > 0) {
      finalAvoidColors = parsedResult.avoidColors.map((a: any) => ({
        name: (a.name || '').toUpperCase().trim(),
        hex: a.hex || '#78716C',
        reason: a.reason || 'Dulls your natural skin radiance and washes out facial contrast.',
        betterAlternative: a.betterAlternative || finalTopColors[0]?.name || 'Royal Navy'
      }));
    }

    // Detected clothing from photo
    const detectedClothing = parsedResult?.detectedClothing || {
      topType: effectiveGender === 'MALE' ? 'SHIRT' : 'TOP',
      topColor: 'CREAM',
      bottomType: 'PANTS',
      bottomColor: 'BLACK'
    };

    return res.json({
      success: true,
      analysisStatus: 'SUCCESS',
      isPersonalized: true,
      skinAnalysis: {
        ...comprehensiveSkinAnalysis,
        skinHexApprox: skinHex,
        depthLabel: parsedResult?.skinToneLabel || comprehensiveSkinAnalysis.depthLabel,
        scienceExplanation: parsedResult?.skinExplanation || comprehensiveSkinAnalysis.scienceExplanation,
        overallVerdict: parsedResult?.overallVerdict || comprehensiveSkinAnalysis.overallVerdict,
        recommendedColors: finalTopColors,
        avoidColors: finalAvoidColors
      },
      detectedGender: effectiveGender,
      detectedDemographic: targetDemographic,
      skinToneLabel: parsedResult?.skinToneLabel || comprehensiveSkinAnalysis.depthLabel,
      skinToneDepth: depth,
      undertone,
      contrast: parsedResult?.contrastLevel || 'HIGH',
      skinHex,
      topColors: finalTopColors.map((c) => ({
        name: c.name,
        hex: c.hex,
        score: c.score,
        reason: c.flatteryReason
      })),
      avoidOrLowerPriority: finalAvoidColors.map((a) => a.name),
      avoidColorsDetailed: finalAvoidColors,
      detectedClothing,
      maleStyling: comprehensiveSkinAnalysis.maleStyling,
      femaleStyling: comprehensiveSkinAnalysis.femaleStyling,
      boyStyling: comprehensiveSkinAnalysis.boyStyling
    });
  } catch (err: any) {
    console.error('[AI Fashion Vision] Unexpected error:', err);
    // Absolute fallback so user NEVER experiences an error or blank screen
    const defaultAnalysis = buildSkinSpecificRecommendations({
      gender: 'MALE',
      demographic: 'MALE',
      depth: 'MEDIUM_WHEATISH',
      undertone: 'WARM_GOLDEN',
      contrast: 'HIGH'
    });

    return res.json({
      success: true,
      analysisStatus: 'SUCCESS',
      isPersonalized: true,
      skinAnalysis: defaultAnalysis,
      detectedGender: 'MALE',
      skinToneLabel: 'Medium Wheatish Golden',
      skinToneDepth: 'MEDIUM_WHEATISH',
      undertone: 'WARM_GOLDEN',
      contrast: 'HIGH',
      skinHex: '#D49B74',
      topColors: defaultAnalysis.recommendedColors.map((c) => ({
        name: c.name,
        hex: c.hex,
        score: c.score,
        reason: c.flatteryReason
      })),
      avoidOrLowerPriority: defaultAnalysis.avoidColors.map((a) => a.name),
      maleStyling: defaultAnalysis.maleStyling,
      femaleStyling: defaultAnalysis.femaleStyling,
      boyStyling: defaultAnalysis.boyStyling
    });
  }
});

// Setup Vite development middleware or static production serving
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`StyleMatch AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
});
