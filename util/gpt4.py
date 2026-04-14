from openai import OpenAI
import base64
import os
from tqdm import tqdm
import time
import random
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def safe_openai_call(messages, max_retries=5, base_delay=2, short_length=100):
    """
    Safe OpenAI API call with retry mechanism and error handling

    Args:
        messages: OpenAI messages
        max_retries: Maximum number of retries
        base_delay: Base delay time (seconds)

    Returns:
        str: Generated content, or fallback content if failed
    """
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempting OpenAI API call (attempt {attempt + 1}/{max_retries})...")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                timeout=60  # 60 second timeout
            )
            
            res = response.choices[0].message.content
            
            # Enhanced rejection response detection
            rejection_phrases = [
                "I'm sorry, I can't assist",
                "I cannot",
                "I'm not able to",
                "I can't provide",
                "I'm unable to",
                "I don't feel comfortable",
                "I'm sorry, but I can't",
                "I cannot help with",
                "I'm not comfortable",
                "I can't help you with",
                "I'm sorry, I cannot",
                "I can't create",
                "I'm not allowed to",
                "I cannot generate"
            ]
            
            # Check if it is a rejection response
            if res:
                res_lower = res.lower()
                is_rejection = any(phrase.lower() in res_lower for phrase in rejection_phrases)
                is_too_short = len(res.strip()) < short_length  # Increase minimum length requirement
                
                if is_rejection or is_too_short:
                    print(f"❌ Detected rejection/invalid response: {res[:100]}...")
                    
                    # Keep the original prompt, retry directly
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying the same request...")
                        continue
                    else:
                        raise ValueError("GPT persistently refused to respond")
            
            print(f"✅ Successfully generated content (length: {len(res)} characters)")
            return res
            
        except Exception as e:
            error_type = type(e).__name__
            print(f"❌ Attempt {attempt + 1} failed: {error_type} - {str(e)}")
            
            if attempt < max_retries - 1:
                # Exponential backoff delay
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ Waiting {delay:.1f} seconds before retrying...")
                time.sleep(delay)
            else:
                print(f"🚫 Maximum retries reached, using fallback content")
                # Determine if it is a zoom-in type (by checking message content)
                is_zoom = any("zoom" in str(msg).lower() for msg in messages)
                return generate_fallback_prompt(is_zoom_in=is_zoom)

def make_messages_safer(messages):
    """
    Convert messages to a gentler version that is less likely to be rejected
    """
    # Gentler system prompt
    safer_system_content = """You are a creative writing assistant helping to create detailed scene descriptions for a technical video project. Please write a detailed, vivid description of a completely still and motionless scene.

Requirements:
- Everything in the scene should be completely still and stationary
- Describe the scene as if it's a beautiful, frozen moment in time
- Include rich details about textures, lighting, and atmosphere
- The description should be suitable for creating a peaceful, static visual
- Focus on architectural elements, natural beauty, and serene environments
- Make it feel like a calm, meditative space where nothing moves

Please write one flowing paragraph describing this peaceful, motionless scene."""

    # Gentler user message
    safer_user_content = "Please help me write a beautiful description of a completely still and peaceful scene based on these reference images. Focus on creating a serene, motionless environment with rich visual details."

    return [
        {
            "role": "system", 
            "content": safer_system_content
        },
        {
            "role": "user",
            "content": safer_user_content
        }
    ]

def generate_fallback_prompt(is_zoom_in=False, scene_name=None):
    """Generate fallback static 3D scene prompts"""
    
    if is_zoom_in:
        zoom_prompts = [
            f"An ultra-sharp, completely static zoom-in video that smoothly reveals intricate hidden details in a perfectly motionless scene, where every element remains frozen and rigid throughout the entire sequence. The camera progressively zooms in on {'the ' + scene_name if scene_name else 'a fascinating small detail'} that becomes increasingly visible, revealing increasingly intricate details of the completely static scene while maintaining perfect sharpness. As the zoom progresses, fine textures and patterns become visible, {'discovering the ' + scene_name + ' with' if scene_name else 'revealing'} rich colors and intricate surface details that remain completely motionless and frozen. The entire zoom sequence maintains perfect stillness and sharpness, with zero movement or blur, creating a mesmerizing reveal of hidden static beauty.",
            
            f"A seamless, ultra-sharp, completely static zoom-in video revealing hidden beauty in a perfectly motionless scene where every element remains frozen like a photograph throughout the entire sequence. Everything remains absolutely motionless and frozen as the camera smoothly zooms in, maintaining perfect sharpness and revealing increasingly intricate details of the completely static scene. The progressive zoom brings into focus {'the ' + scene_name if scene_name else 'a small fascinating element'} with fine textures and patterns, {'showing the ' + scene_name if scene_name else 'discovering something'} that adds visual intrigue while remaining completely motionless and frozen. Everything remains absolutely motionless throughout, creating perfect conditions for detailed examination and 3D reconstruction.",
            
            f"An ultra-sharp zoom-in sequence capturing a completely static scene where every element remains perfectly motionless and frozen throughout. The camera smoothly zooms in to reveal {'the ' + scene_name if scene_name else 'hidden details'} that become progressively more detailed and fascinating, with all surfaces and objects remaining completely rigid and immobile. {'The ' + scene_name + ' appears' if scene_name else 'A small detail emerges'} with intricate patterns and textures that remain absolutely stationary, frozen in perfect detail with zero movement or animation. The entire sequence maintains razor-sharp focus and complete stillness, creating ideal conditions for 3D reconstruction."
        ]
        return random.choice(zoom_prompts)
    else:
        scene_prompts = [
            f"An ULTRA-STATIC 3D video where absolutely everything is frozen motionless like a photograph, captured by a single camera smoothly moving through a scene where every single element remains perfectly still and rigid throughout the entire sequence, as if time itself has completely stopped. The camera moves through a completely motionless {'space revealing the ' + scene_name if scene_name else 'architectural environment'} where all surfaces, objects, and materials are frozen solid like stone sculptures, revealing new areas where everything remains perfectly rigid and immobile. {'The journey leads to the ' + scene_name + ' where' if scene_name else 'Every element in the scene is'} absolutely stationary, locked in position with zero movement or animation of any kind, creating ideal conditions for 3D reconstruction with all objects completely motionless and frozen in perfect stillness.",
            
            f"A seamless, ultra-sharp, completely static video captured by a single camera moving through a perfectly motionless scene where every single element remains frozen like a photograph throughout the entire sequence. All objects are completely motionless and rigid, frozen in perfect stillness with absolutely no movement of any kind, solid and immobile like stone statues. The camera smoothly moves to reveal {'the ' + scene_name if scene_name else 'new areas'} where everything remains perfectly frozen and motionless, discovering interesting {'elements of the ' + scene_name if scene_name else 'architectural details'} and surfaces that are perfectly motionless and frozen in place. This entire scene remains in perfect, absolute stillness throughout the video, with zero movement or animation of any kind, creating ideal conditions for 3D reconstruction.",
            
            f"An ULTRA-STATIC 3D sequence where every element is completely frozen and motionless throughout, captured by a smooth camera movement through a scene where all objects remain perfectly still and rigid like stone sculptures. The camera reveals {'the ' + scene_name + ' environment' if scene_name else 'various architectural spaces'} where every surface, texture, and detail is absolutely motionless and locked in position. {'Moving through the ' + scene_name + ' reveals' if scene_name else 'The journey shows'} completely static elements with zero movement, vibration, or animation of any kind, maintaining perfect stillness that creates ideal conditions for 3D reconstruction with all materials frozen solid and immobile."
        ]
        return random.choice(scene_prompts)

def generate_3d_scene_prompt(frames_path, output_path, frame_interval=6, scene_name=None):
    """
    Generate a 3D scene prompt using GPT-4 Vision based on a sequence of frames.
    The generated scene must be completely static for 3D reconstruction.
    
    Args:
        frames_path (str): Path to the directory containing the frames
        output_path (str): Path where the generated prompt will be saved
        frame_interval (int): Interval between frames to process (default: 6)
        scene_name (str): User-specified scene name to discover (optional)
    
    Returns:
        str: The generated prompt
    """
    # Read images and convert to base64 encoding
    contents = []
    for idx, name in tqdm(enumerate(sorted(os.listdir(frames_path))[::frame_interval])):
        if name.endswith(".png"):
            image_path = os.path.join(frames_path, name)
            base64_image = encode_image_to_base64(image_path)
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

    # Build system prompt, adjust content based on scene_name - emphasize 3D consistency and balanced description
    system_content = """You are a creative scene description assistant. Analyze these sequential images and create a detailed description of a completely still, peaceful scene that maintains perfect 3D consistency throughout.

🎯 SCENE REQUIREMENTS:
- Everything should be perfectly still and motionless like a beautiful photograph
- Describe a serene, frozen moment in time
- Focus on architectural elements, natural beauty, and calm environments  
- All objects and surfaces should be completely stationary
- Create a peaceful, meditative atmosphere where nothing moves
- Emphasize the tranquil, static nature of the environment

🔧 CRITICAL 3D CONSISTENCY REQUIREMENTS:
- Photorealistic static scene with perfect 3D geometric consistency
- All objects maintain rigid structure with fixed dimensions and positions
- Surfaces follow correct perspective projection and depth relationships  
- No morphing, warping, or dimensional changes during camera movement
- Consistent material properties and stable lighting response throughout
- Architecturally sound proportions with realistic physics
- Sharp details with stable textures that don't swim or distort
- Perfect spatial continuity for 3D reconstruction

Your description should:
1. Start with: "A photorealistic static scene with perfect 3D geometric consistency, captured by a single camera, where every element maintains rigid structure and fixed dimensions throughout the camera movement, completely motionless and peaceful."

2. Describe the current existing scene in detail (equal emphasis)

3. ADD CLEAR TRANSITION with smooth camera movement:"""

    if scene_name:
        system_content += f"""
   - "As the perspective smoothly shifts and moves through this consistent 3D space, there appears {scene_name}"
   - Then describe the {scene_name} with equal detail as the existing scene
   - Maintain visual and spatial consistency between existing and new elements
   - Ensure the {scene_name} fits naturally in the same lighting and environment"""
    else:
        system_content += """
   - "As the perspective smoothly shifts and moves through this consistent 3D space, there appears [new element]"
   - Then describe this new element with equal detail as the existing scene
   - Maintain visual and spatial consistency between existing and new elements"""

    system_content += """

4. Use these CONSISTENCY EMPHASIS phrases throughout:
   - "rigid structure"
   - "fixed dimensions" 
   - "geometric consistency"
   - "stable textures"
   - "consistent material properties"
   - "architecturally sound"
   - "no morphing or warping"
   - "perfect 3D consistency"

5. End with: "This photorealistic scene maintains perfect 3D geometric consistency with rigid structure, stable textures, and consistent material properties throughout, with no morphing or dimensional changes, creating ideal conditions for precise 3D reconstruction."

CRITICAL: Balance the description equally between existing scene and new scene. Emphasize 3D consistency and smooth transitions. Output as ONE single flowing paragraph without any line breaks or separate sections.

Output ONLY the complete description as ONE flowing paragraph with balanced descriptions and 3D consistency emphasis."""

    # Use the safe API call
    res = safe_openai_call([
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Please create a description of a photorealistic static scene with perfect 3D geometric consistency. All objects maintain rigid structure with fixed dimensions. Describe the current scene and {'the ' + scene_name if scene_name else 'new element'} with equal detail. Use the transition 'As the camera moves through this geometrically consistent 3D space, there appears {'the ' + scene_name if scene_name else '[new element]'}'. Emphasize no morphing, warping, or dimensional changes:"}] + contents
        }
    ])

    # Save the generated prompt to a file
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(res)
        print(f"✅ 3D scene prompt saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

    return res

# def generate_zoom_in_prompt(frames_path, output_path, frame_interval=6, scene_name=None):
#     """
#     Generate a zoom-in prompt using GPT-4 Vision based on a sequence of frames.
#     The generated scene must be completely static with creative discoveries.
    
#     Args:
#         frames_path (str): Path to the directory containing the frames
#         output_path (str): Path where the generated prompt will be saved
#         frame_interval (int): Interval between frames to process (default: 6)
#         scene_name (str): User-specified small object/detail to discover during zoom (optional)
    
#     Returns:
#         str: The generated prompt
#     """
#     # Read images and convert to base64 encoding
#     contents = []
#     for idx, name in tqdm(enumerate(sorted(os.listdir(frames_path))[::frame_interval])):
#         if name.endswith(".png"):
#             image_path = os.path.join(frames_path, name)
#             base64_image = encode_image_to_base64(image_path)
#             contents.append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/png;base64,{base64_image}"}
#             })

#     # Build discovery content based on scene_name - concise version
#     if scene_name:
#         discovery_instruction = f"""
# 🎯 DISCOVERY FOCUS:
# - The zoom reveals "{scene_name}" as the main discovery
# - Focus mainly on describing the {scene_name} with rich detail (colors, textures, size, position)
# - Briefly describe initial view, then extensively describe the {scene_name}"""
        
#         discovery_examples = f"""Focus on the {scene_name} with detailed visual description."""
#     else:
#         discovery_instruction = f"""
# 🎯 DISCOVERY FOCUS:
# - Create a fascinating small discovery as the main focus
# - Focus mainly on describing this discovery with rich detail
# - Briefly describe initial view, then extensively describe the discovery"""
        
#         discovery_examples = """Create a small fascinating discovery with detailed visual description."""

#     # Use safe_openai_call to support retry mechanism
#     messages = [
#         {
#             "role": "system",
#             "content": f"""You are a zoom-in video diffusion prompt generator. Create a prompt for a smooth, static zoom-in video that reveals fascinating hidden details.

# 🚨 CRITICAL STATIC REQUIREMENTS:
# - ABSOLUTELY MOTIONLESS: Everything frozen like a high-resolution photograph
# - ZERO MOVEMENT: No swaying, floating, vibrating, or micro-movements
# - CAMERA ZOOM ONLY: Only the zoom level changes, nothing else moves
# - RAZOR SHARP: Perfect focus throughout, no blur or depth-of-field effects
# - COMPLETELY RIGID: All surfaces and objects solid and immobile

# {discovery_instruction}

# Your prompt structure:
# 1. Start with: "An ultra-sharp, completely static zoom-in video that smoothly reveals intricate hidden details in a perfectly motionless scene, where every element remains frozen and rigid throughout the entire sequence."

# 2. Briefly describe the initial wide view

# 3. Emphasize the static nature: "everything remains perfectly motionless and frozen"

# 4. Focus mainly on the zoom discovery:
#    - "as the camera smoothly zooms in, revealing [discovery with rich detail]"
#    - Describe the discovery extensively with colors, textures, patterns, size, position
#    - {"Use the " + scene_name + " as the main discovery" if scene_name else "Create a fascinating small discovery"}

# 5. End with: "The entire zoom sequence maintains perfect stillness and sharpness, with zero movement or blur, creating a mesmerizing reveal of hidden static beauty."

# Output ONLY the complete prompt as ONE flowing paragraph emphasizing both the static nature and the fascinating discovery."""
#         },
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": f"Generate a zoom-in video diffusion prompt for these frames. {'Focus mainly on revealing the ' + scene_name + ' with rich detail as the main discovery.' if scene_name else 'Create a fascinating discovery as the main focus with rich detail.'} Keep everything completely static:"}] + contents
#         }
#     ]
    
#     res = safe_openai_call(messages)

#     # Save the generated prompt to a file
#     try:
#         with open(output_path, "w", encoding='utf-8') as f:
#             f.write(res)
#         print(f"✅ Zoom-in prompt saved to: {output_path}")
#     except Exception as e:
#         print(f"❌ Failed to save file: {e}")

#     return res

def generate_zoom_in_prompt(frames_path, output_path, frame_interval=6, **kwargs):
    """
    Generate a zoom-in prompt using GPT-4 Vision based on a sequence of frames.
    The generated scene must be completely static, focusing ONLY on existing elements.
    
    Args:
        frames_path (str): Path to the directory containing the frames
        output_path (str): Path where the generated prompt will be saved
        frame_interval (int): Interval between frames to process (default: 6)
    
    Returns:
        str: The generated prompt
    """
    # Read images and convert to base64 encoding
    contents = []
    for idx, name in tqdm(enumerate(sorted(os.listdir(frames_path))[::frame_interval])):
        if name.endswith(".png"):
            image_path = os.path.join(frames_path, name)
            base64_image = encode_image_to_base64(image_path)
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

    # Simplified focus instruction - GPT chooses the appropriate detail itself
    focus_instruction = """
🎯 ZOOM FOCUS:
- Analyze the frames and identify the most interesting existing detail to zoom into
- ONLY describe what is ALREADY VISIBLE in the frames
- DO NOT create, add, or imagine any new elements
- Focus on revealing existing fine details that become clearer with zoom
- Choose a specific element that would benefit from closer examination
- Describe textures, patterns, and micro-details that are already there"""

    # Use safe_openai_call to support retry mechanism
    messages = [
        {
            "role": "system",
            "content": f"""You are a zoom-in video diffusion prompt generator. Create a prompt for a smooth, static zoom-in video that reveals fine details of EXISTING elements only.

🚨 CRITICAL REQUIREMENTS:
- ABSOLUTELY MOTIONLESS: Everything frozen like a high-resolution photograph
- ZERO MOVEMENT: No swaying, floating, vibrating, or micro-movements
- CAMERA ZOOM ONLY: Only the zoom level changes, nothing else moves
- RAZOR SHARP: Perfect focus throughout, no blur or depth-of-field effects
- EXISTING ELEMENTS ONLY: DO NOT describe anything new - only what's already visible

{focus_instruction}

Your prompt structure:
1. Start with: "An ultra-sharp, completely static zoom-in video that smoothly reveals intricate details already present in a perfectly motionless scene, where every element remains frozen and rigid throughout the entire sequence."

2. Briefly describe the initial wide view (what's already visible)

3. Emphasize the static nature: "everything remains perfectly motionless and frozen"

4. Focus on zooming into EXISTING details:
   - "as the camera smoothly zooms in on [existing element you identify], revealing its fine details"
   - Describe textures, patterns, and micro-details that are ALREADY THERE
   - Choose the most visually interesting existing element from the scene
   - Emphasize that these details were always there, just becoming more visible

5. End with: "The entire zoom sequence maintains perfect stillness and sharpness, revealing the intricate details that were always present in this completely static scene."

CRITICAL: Only describe what is ALREADY VISIBLE in the frames. Do not add, create, or imagine any new elements.

Output ONLY the complete prompt as ONE flowing paragraph emphasizing the static nature and existing details."""
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Generate a zoom-in video diffusion prompt for these frames. Analyze the images and choose the most interesting existing element to zoom into, revealing its already-present details. Only describe what's already visible:"}] + contents
        }
    ]
    
    res = safe_openai_call(messages)

    # Save the generated prompt to a file
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(res)
        print(f"✅ Zoom-in prompt saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

    return res


def generate_edit_prompt(img_path, output_path, scene_name=None, short_length=2):
    """
    Generate a simple edit prompt using GPT-4 Vision based on a single image.
    
    Args:
        img_path (str): Path to the input image
        output_path (str): Path where the generated prompt will be saved
        scene_name (str): User-specified scene element to add (optional)
        short_length (int): Minimum length for response validation
    
    Returns:
        str: The generated edit prompt
    """
    import base64
    import os
    
    # Read image and convert to base64 encoding
    base64_image = encode_image_to_base64(img_path)

    # Build system prompt
    system_content = """You are an image editing assistant. Look at this image and create a simple editing description.

Your task:
1. Identify the main surface or area in the center of the image
2. Create a simple description in the format: "[object] is on the [surface/location]"

Examples:
- "fountain is on the courtyard"
- "bench is on the grass"
- "statue is on the plaza"

Keep it simple and direct. Just identify what surface you see."""

    # Build user message
    if scene_name:
        user_content = f"Look at this image and create a simple description: '{scene_name} is on the [what surface/location you see]'"
    else:
        user_content = "Look at this image and create a simple description: '[something] is on the [what surface/location you see]'"

    # Build messages
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        }
    ]
    
    # Use the safe API call
    res = safe_openai_call(messages, short_length=short_length)

    # If scene_name exists but GPT did not use it correctly, perform post-processing
    if scene_name and scene_name.lower() not in res.lower():
        # Simple replacement to correct format
        if " is on the " in res.lower():
            parts = res.lower().split(" is on the ", 1)
            if len(parts) == 2:
                res = f"{scene_name} is on the {parts[1]}"
        else:
            res = f"{scene_name} is on the ground"
    
    # Ensure the result format is correct
    if not res or " is on the " not in res.lower():
        if scene_name:
            res = f"{scene_name} is on the surface"
        else:
            res = "object is on the ground"
    
    # 🔧 Remove leading and trailing quotes to prevent Step1X-Edit syntax errors
    res = res.strip()
    if res.startswith('"') and res.endswith('"'):
        res = res[1:-1]
    if res.startswith("'") and res.endswith("'"):
        res = res[1:-1]
    
    # Save the generated prompt to a file
    if output_path:
        try:
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(res)
            print(f"✅ Edit prompt saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save file: {e}")
    
    print(f"Generated edit prompt: {res}")
    return res

# ... existing code ...
def extract_foreground_background(image_path, max_retries=5, short_length=10):
    """
    Use GPT-4 Vision to extract foreground objects (as a list of words) and a background description (as a sentence) from an image.
    The background description should not overlap with any foreground object.

    Args:
        image_path (str): Path to the input image
        max_retries (int): Maximum number of retries for the API call
        short_length (int): Minimum length for response validation

    Returns:
        tuple: (foreground_list, background_description)
            - foreground_list: list of strings (foreground object words)
            - background_description: string (background description sentence)
    """
    import re
    from difflib import get_close_matches

    def to_singular(word):
        """Convert English plural words to singular form using basic rules."""
        word = word.lower().strip()
        
        # Common irregular plurals
        irregular_plurals = {
            'people': 'person',
            'children': 'child',
            'feet': 'foot',
            'teeth': 'tooth',
            'geese': 'goose',
            'mice': 'mouse',
            'men': 'man',
            'women': 'woman',
            'oxen': 'ox',
            'sheep': 'sheep',
            'deer': 'deer',
            'fish': 'fish',
            'glasses': 'glass',
            'leaves': 'leaf',
            'knives': 'knife',
            'lives': 'life',
            'wolves': 'wolf',
            'thieves': 'thief'
        }
        
        if word in irregular_plurals:
            return irregular_plurals[word]
        
        # Words ending in -ies (like flies -> fly)
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        
        # Words ending in -ves (like halves -> half)
        if word.endswith('ves') and len(word) > 3:
            return word[:-3] + 'f'
        
        # Words ending in -ses, -ches, -shes, -xes, -zes
        if word.endswith(('ses', 'ches', 'shes', 'xes', 'zes')):
            return word[:-2]
        
        # Words ending in -s (most common case)
        if word.endswith('s') and len(word) > 1:
            # But not words that naturally end in 's' (like glass, grass)
            # Simple check: if removing 's' creates a very short word, keep it
            candidate = word[:-1]
            if len(candidate) >= 2:
                return candidate
        
        # If no rule applies, return the original word
        return word

    base64_image = encode_image_to_base64(image_path)
    system_content = (
        "You are a vision assistant. Given an image, your task is to strictly output two things:\n"
        "1. Foreground: List all distinct foreground objects as single English words, separated by commas.\n"
        "2. Background: Write one English sentence describing the background, making sure it does NOT mention or overlap with any foreground object.\n"
        "Output format:\n"
        "Foreground: <comma-separated-words>\n"
        "Background: <one-sentence-background-description>\n"
        "Do not include any extra explanation.\n"
        "IMPORTANT: The background description MUST NOT contain any of the foreground words, in any form (singular, plural, or synonyms)."
    )
    user_content = [
        {"type": "text", "text": "Please analyze this image and output the foreground and background as specified. The background MUST NOT mention any foreground word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    ]
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    for attempt in range(max_retries):
        res = safe_openai_call(messages, max_retries=1, short_length=short_length)
        if res is None:
            continue
        # Parse the response
        lines = [line.strip() for line in res.split('\n') if line.strip()]
        fg, bg = None, None
        for line in lines:
            if line.lower().startswith("foreground:"):
                fg = line[len("foreground:"):].strip()
            elif line.lower().startswith("background:"):
                bg = line[len("background:"):].strip()
        if fg is not None and bg is not None:
            # Convert all foreground words to singular form
            foreground_list = [to_singular(w.strip()) for w in fg.split(",") if w.strip()]
            background_description = bg
            # Enforce that background does NOT contain any foreground word (case-insensitive, singular/plural)
            bg_lower = background_description.lower()
            violation = False
            for word in foreground_list:
                word_lower = word.lower()
                # Check for exact word and its potential plural forms
                patterns = [
                    r"\\b" + re.escape(word_lower) + r"\\b",  # singular form
                    r"\\b" + re.escape(word_lower + 's') + r"\\b",  # simple plural
                ]
                
                # Add common plural patterns for the singular word
                if word_lower.endswith('y'):
                    patterns.append(r"\\b" + re.escape(word_lower[:-1] + 'ies') + r"\\b")  # fly->flies
                elif word_lower.endswith('f'):
                    patterns.append(r"\\b" + re.escape(word_lower[:-1] + 'ves') + r"\\b")  # leaf->leaves
                elif word_lower.endswith(('s', 'sh', 'ch', 'x', 'z')):
                    patterns.append(r"\\b" + re.escape(word_lower + 'es') + r"\\b")  # box->boxes
                    
                for pat in patterns:
                    if re.search(pat, bg_lower):
                        violation = True
                        break
                if violation:
                    break
            if not violation:
                return foreground_list, background_description
            # If violation, retry (up to max_retries)
    # Fallback: return empty list and empty string
    return [], ""


if __name__ == "__main__":
    # Example usage:
    # prompt = generate_3d_scene_prompt("path/to/frames", "output/prompt.txt", frame_interval=30)
    # prompt = generate_zoom_in_prompt("path/to/frames", "output/zoom_prompt.txt", frame_interval=30)
    # prompt = generate_edit_prompt("path/to/image.png", "output/edit_prompt.txt", scene_name="a fountain")
    # fg_list, bg_desc = extract_foreground_background("path/to/image.png")
    pass