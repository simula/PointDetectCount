from datasets import load_dataset
import random
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Create instruction formatted JSONL files")
parser.add_argument("--image-dir", default="MedMultiPoints-images",
                    help="Directory containing dataset images")
parser.add_argument("--output", default="kvasir_valid.jsonl",
                    help="Output JSONL file name")
args = parser.parse_args()

ds = load_dataset("SushantGautam/kvasir-points")
random.seed(42)

GENERAL_PROMPTS_V1 = {
  "pointing": [
    "Point to {label}.\nPlease say 'This isn't in the image.' if it is not in the image.",
    "Point to all occurrences of \"{label}s\"",
    "Point to any {label}s in the image",
    "Point to any {label}s in the image.",
    "Point: Where are the {label}s?",
    "Show me where the {label}s are",
    "Can you show me where the {label}s are?",
    "Show me where the {label}s are",
    "Show me where a {label} is",
    "Show me where a {label} is.",
    "If there are any {label}s in the image, show me where they are.",
    "Where are the {label}s?",
    "Generate a list of points showing where the {label}s are.",
    "Find the \"{label}\".",
    "Find a \"{label}\".",
    "Locate all {label}s.",
    "Locate an {label}.",
    "Locate a {label}.",
    "Locate every {label}.",
    "Locate {label}.",
    "Locate the {label}.",
    "Object: {label}\nInstruction: Point to the object.",
    "Find {label}.",
    "Find {label}.",
    "Point to every {label}",
    "Find any {label}s in the picture",
    "Find the {label}.",
    "Find any {label}s",
    "Point to a {label}",
    "Point to an {label}",
    "Look for {label}s in the image and show me where they are.",
    "Help me find an object in the image by pointing to it.\nObject: {label}.",
    "I am looking for {label}s. Where can they be found in the image?",
    "Can you see any {label}s in the image? Point to them.",
    "Point out each {label} in the image.",
    "Point out every {label} in the image.",
    "Point to the {label} in the image.",
    "Locate each {label} in the image.",
    "Can you point out all {label}s in this image?",
    "Please find {label}s and show me where they are.",
    "If there are any {label}s present, indicate their positions.",
    "If there is a {label} present, indicate its position.",
    "show me all visible {label}s"
  ],
    "counting": [
        "How many {label} are there?",
        "How many {label}?",
        "How many {label}.",
        "how many {label}.",
        "how many {label}?",
        "How many {label} are there in the image?",
        "Tell me how many {label} there are",
        "Tell me how many {label} there are and point to them.",
        "how many {label}",
        "Tell me where each {label} is.",
        "Tell me how many {label} are in the image",
        "count {label}",
        "count every {label}",
        "count each {label}",
        "count {label}.",
        "Count the {label}.",
        "How many {label} do you see?",
        "How many {label} are visible?",
        "Count all the {label}",
        "how mmny {label}?",
        "Count every {label} in the picture.",
        "Count all the {label}",
        "Count each {label}",
        "Point to and count the {label} in the picture.",
        "Point and count {label}",
        "Point to every {label}",
        "Locate the {label} and count them",
        "Locate every {label} and count them",
        "Find all the {label}. How many are there?",
        "Find each {label}. How many are there?",
        "Point at {label} and then tell me the count.",
        "What is the total number of {label} in the image?",
        "In all the picture, how many {label} are there?",
        "Point at the {label} and then count them.",
        "Point to all the visible {label} output the total count.",
        "Point to all the {label} visible and output the total count. \nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\" and output the total count.",
        "Show me where the {label} are and output the total count.",
        "Where are the {label}? How many are there?",
        "Generate list of points showing where the {label} are and output the total count.",
        "Object: {label}\nInstruction: Point to the object and output the total count.",
        "find any {label} in the picture and output the total count.",
        "Can you see any {label} in the image? Point to them and output the total count.",
        "Can you point out all {label} in this image? How many are there?",
        "If there are any {label} present, indicate their positions and output the total count.",
        "How many {label} are there in the image? Point to them and output the total count.",
        "How many {label} are there in the image?",
        "Give me the count of {label} in the image.",
        "How many {label} are visible in the image?",
        "How many {label} are there?",
        "In the image, how many {label} are there?",
        "Can you count the number of {label} in the image?",
        "Can you count every {label} in the picture?",
        "Can you see any {label} in the image? How many are there?",
        "Are there any {label} in the image? How many are there?",
        "If you see any {label} in the image, give me the count. Otherwise, say 'This isn't in the image.'",
        "Object: {label}\nInstruction: How many are there?",
    ],
    "cnt_and_point": [
        "Count the {label} in the image, then point to them.",
        "How many {label} are there? Point to them.",
        "Count every {label} in the picture, then point to them.",
        "Locate the {label} and count them, then point to them.",
        "Find all the {label}. How many are there? Point to them.",
        "Find each {label}. How many are there? Point to them.",
        "Point to and count the {label} in the picture.",  
        "Point to all {label} in the image and then count them.",  ## added from here
        "Indicate the position of every {label} and provide the total count.",
        "Mark each {label} with a point, then output how many there are.",
        "Locate all {label} in the image by pointing to them and then counting them.",
        "Point out every {label} you see in the image, then tell me the total number.",
        "Identify all occurrences of {label} by marking them, and then provide the count.",
        "Show me each {label} by pointing at them, and then specify the total count.",
        "Indicate each {label} in the image with a point and then return the overall count.",
        "Point to each instance of {label} and then output their total number.",
        "Mark all visible {label} in the image with a point and then provide the count."
    ],
    "bounding": [
        "Draw a bounding box around {label}.",
        "Locate and mark {label} with a bounding box.",
        "Identify the {label} in the image and provide its bounding box coordinates.",
        "Provide the coordinates of a bounding box that encloses the {label}.",
        "Enclose {label} in a bounding box. Format: [xx, yy, xx, yy].",
        "Find the {label} in the image and output a bounding box around it.",
        "Outline the {label} using a bounding box.",
        "Return a bounding box that encloses {label}.",
        "Mark the {label} with a rectangle and supply the bounding box coordinates.",
        "Provide a bounding box that tightly encloses the {label}.",
        "Generate bounding box coordinates for the {label}.",
        "Specify the bounding box for the {label} in the image.",
        "Identify {label} and output its bounding box in [xx, yy, xx, yy] format.",
        "Return the bounding box of the {label}.",
        "Mark the object {label} with a bounding box and return its coordinates.",
        "Locate the {label} and mark it with a bounding box in the image.",
        "Outline the area containing {label} by returning a bounding box.",
        "Find the region corresponding to {label} and provide the bounding box.",
        "Detect the {label} and enclose it in a bounding box with coordinates.",
        "Return the rectangular bounding box that covers {label} in the image."
    ],
    "count_and_box": [
        "Count all {label} in the image and provide their bounding boxes.",
        "Find all {label} in the image, output the count and their bounding boxes.",
        "Count every {label} and return a list of bounding boxes.",
        "Identify all {label} in the image, count them, and provide bounding box coordinates for each.",
        "Detect all {label} in the image, and for each, output its bounding box. Also, provide the total count.",
        "Return both the count and the bounding boxes of all {label} in the image.",
        "Provide the total number of {label} and for each, supply a bounding box.",
        "Locate all instances of {label}, count them, and return their bounding boxes.",
        "Count the {label} and enclose each in a bounding box.",
        "Output the total count of {label} in the image along with bounding boxes for each.",
        "Draw bounding boxes around every {label} and then provide the total count.",
        "Outline all instances of {label} with bounding boxes, then return the count.",
        "Return bounding boxes for each {label} in the image and the total count.",
        "Mark all {label} with a bounding box, then count how many are present.",
        "Provide bounding box coordinates for each {label} and then specify the total count.",
        "Detect all {label} in the image with bounding boxes, and then output their count.",
        "Mark each {label} with a bounding box and return the number of instances.",
        "Find all {label} in the image, draw bounding boxes around them, and then count them.",
        "Output a list of bounding boxes for every {label} and the overall count.",
        "Identify each {label} by its bounding box, and then provide the total count."
    ]
}


# import cv2
# def molmo_coords(coords, w, h):
#     return coords[0] / w * 100, coords[1] / h * 100


ran_label = lambda x: random.choice({
    'normal': ["normal sperm", "sperm"],
    'pinhead': ["pinhead sperm", "pinhead"],
    'cluster': ["sperm cluster", "cluster"],
    'instrument': ["instrument"],
    'polyps': ["polyp"]
    }.get(x, [x]))

jsonl=[]

#### pointing task 
for idx, data in enumerate(ds['train']):
    # if idx > 10:
    #     break
    # print(data)
    points = data['points']

    image_name = os.path.join(args.image_dir, data['image_sha256'] + '.png')
    # if not os.path.exists(image_name):
    #     data['image_data'].save(image_name)
    # h, w = cv2.imread(image_name).shape[:2]
    # mol_points = [molmo_coords(p, h, w) for p in points]
    # s = f"""<points {' '.join(f'x{i+1}="{x:.1f}" y{i+1}="{y:.1f}"' for i, (x, y) in enumerate(mol_points))} alt="{label}">{label}</points>""" 
    fmt_json = lambda d: "```json\n" + json.dumps(d) + "\n```"
    mk_ent   = lambda typ, lbl, cnt, tsk: {
        "messages": [
            {"role": "user", "content": "<image> " + random.choice(GENERAL_PROMPTS_V1[typ]).format(label=lbl), 
             "task": tsk, "source": data['classification']},
            {"role": "assistant", "content": cnt}
        ],
        "images": [image_name]
    }
    p_lbl, b_lbl, c_lbl, cp_lbl = [ran_label(data['label']) for _ in range(4)]
    ent_pt    = mk_ent("pointing",    p_lbl, fmt_json([{"point_2d": [round(c, 1) for c in pt], "label": p_lbl} for pt in points]), "pointing")
    ent_bb    = mk_ent("bounding",    b_lbl, fmt_json([{"bbox_2d": bb, "label": b_lbl} for bb in data['bbox']]), "bounding")
    ent_cnt   = mk_ent("counting", c_lbl, fmt_json({"counts": data['count'], "label": c_lbl}), "counting")
    ent_cntpt = mk_ent("cnt_and_point", cp_lbl, fmt_json({"counts": data['count'], "point_2d": [[round(c, 1) for c in pt]  for pt in points], "label": c_lbl, }), "cnt_and_point")
    ent_cnt_bb = mk_ent("count_and_box", cp_lbl, fmt_json({"counts": data['count'], "bbox_2d": [bb for bb in data['bbox']], "label": c_lbl, } ), "count_and_box")
    jsonl.extend([ent_pt, ent_bb, ent_cnt, ent_cntpt, ent_cnt_bb])

with open(args.output, "w") as f:
    f.writelines(json.dumps(entry) + "\n" for entry in jsonl)


# pointing: :[{\"point_2d\": [x,y], \"label\": p_lbl},.., {..}].", "task": "pointing", 
# bounding: :[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": b_lbl},.., {..}].", "task": "bounding",
# counting: :{\"count\": count, \"label\": c_lbl}.", "task": "counting",
# cnt_and_point: :{\"count\": count, \"point_2d\": [[x,y],.., [x,y]], \"label\": c_lbl, }.", "task": "cnt_and_point",
# count_and_box: :{\"count\": count, \"bbox_2d\": [[x1,y1,x2,y2],.., [x1,y1,x2,y2]], \"label\": c_lbl, }.", "task": "count_and_box",
#
