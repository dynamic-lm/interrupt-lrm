##############################
# MATH Prompt Templates
##############################

INTERVENE_MATH_SYS_PROMPT = """
You will be given a math problem.

# Task
* Revise the problem by modifying **four** specifications.
  - Other than the four changes, directly copy the original problem text character by character.
  - If there are less than four specifications in the input problem, modify the maximum number of specifications possible.
  - Use the same math formatting as the original problem (e.g., notation, style, backslashes, dollar signs, etc.)

# Examples
* You will be given a set of example input-output pairs to help you understand the task.

# Output
* Output the revised problem only. Do not include any other text in your output.
""".strip()

INTERVENE_MATH_PROMPT = """
# Examples
--------------------------------------------------
INPUT 1:
An airport has only 2 planes that fly multiple times a day. Each day, the first plane goes to Greece for three-quarters of its flights, and the remaining flights are split equally between flights to France and flights to Germany. The other plane flies exclusively to Poland, and its 44 trips only amount to half the number of trips the first plane makes throughout each day. How many flights to France does the first plane take in one day?

OUTPUT 1:
An airport has only 2 planes that fly multiple times a day. Each day, the first plane goes to Greece for one-quarters of its flights, and the remaining flights are split equally between flights to France, Spain, and Germany. The other plane flies exclusively to Poland, and its 22 trips only amount to one third the number of trips the first plane makes throughout each day. How many flights to France does the first plane take in one day?

INPUT 2:
Let $F_1 = (10,2)$ and $F_ 2= (-16,2).$  Then the set of points $P$ such that\n\\[|PF_1 - PF_2| = 24\\]form a hyperbola.  The equation of this hyperbola can be written as\n\\[\\frac{{(x - h)^2}}{{a^2}} - \\frac{{(y - k)^2}}{{b^2}} = 1.\\]Find $h + k + a + b.$

OUTPUT 2:
Let $F_1 = (5,5)$ and $F_ 2= (-8,8).$  Then the set of points $P$ such that\n\\[|PF_1 - PF_2| = 12\\]form a hyperbola.  The equation of this hyperbola can be written as\n\\[\\frac{{(x - h)^2}}{{a^2}} - \\frac{{(y - k)^2}}{{b^2}} = 1.\\]Find $h.$

INPUT 3:
Let ${{\\triangle ABC}}$ be a right triangle with $\\angle A = 90^\\circ$ and $BC = 38.$ There exist points $K$ and $L$ inside the triangle such\\[AK = AL = BK = CL = KL = 14.\\]The area of the quadrilateral $BKLC$ can be expressed as $n\\sqrt3$ for some positive integer $n.$ Find $n.$

OUTPUT 3:
Let ${{\\triangle ABC}}$ be a right triangle with $\\angle A = 90^\\circ$ and $BC = 19.$ There exist points $K$ and $L$ inside the triangle such\\[AK = BK = KL = 28.\\]Find the area of the quadrilateral $BKLC$.
--------------------------------------------------

INPUT:
{problem}

OUTPUT:

""".strip()

INTERVENE_MATH_UPDATE_SYS_PROMPT = """
You will be given two math problems -- PROBLEM A and PROBLEM B.
PROBLEM B is the modified version of PROBLEM A.

# Task
Generate a short instructional update describing the change from PROBLEM A to PROBLEM B.
* Do not mention PROBLEM A or PROBLEM B in the update.
* If there are multiple changes, please make sure each change is clearly described in the update.

# Examples
* You will be given a set of example input-output pairs to help you understand the task.
* Follow the format (style, language, length, etc.) of the outputs in the provided examples.

# Output
* Output the update only. Do not include any other text in your output.
""".strip()

INTERVENE_MATH_UPDATE_PROMPT = """
# Examples
--------------------------------------------------
Input 1:
PROBLEM A
An airport has only 2 planes that fly multiple times a day. Each day, the first plane goes to Greece for one-quarters of its flights, and the remaining flights are split equally between flights to France, Spain, and Germany. The other plane flies exclusively to Poland, and its 22 trips only amount to one third the number of trips the first plane makes throughout each day. How many flights to France does the first plane take in one day?
PROBLEM B
An airport has only 2 planes that fly multiple times a day. Each day, the first plane goes to Greece for three-quarters of its flights, and the remaining flights are split equally between flights to France and flights to Germany. The other plane flies exclusively to Poland, and its 44 trips only amount to half the number of trips the first plane makes throughout each day. How many flights to France does the first plane take in one day?

Output 1:
The trips to Greece for the first plane correspond to 75% of its total flights and the remaining flights are equally split between France and Germany. The total flights of the second plane, 22, are half the amount of flights of the first plane.

Input 2:
PROBLEM A
Let $F_1 = (5,5)$ and $F_ 2= (-8,8).$  Then the set of points $P$ such that\n\\[|PF_1 - PF_2| = 12\\]form a hyperbola.  The equation of this hyperbola can be written as\n\\[\\frac{{(x - h)^2}}{{a^2}} - \\frac{{(y - k)^2}}{{b^2}} = 1.\\]Find $h.$
PROBLEM B
Let $F_1 = (10,2)$ and $F_ 2= (-16,2).$  Then the set of points $P$ such that\n\\[|PF_1 - PF_2| = 24\\]form a hyperbola.  The equation of this hyperbola can be written as\n\\[\\frac{{(x - h)^2}}{{a^2}} - \\frac{{(y - k)^2}}{{b^2}} = 1.\\]Find $h + k + a + b.$

Output 2:
The coordinates of the first point are (10,2) and the distance between the point should be 24. F_2 = (-16,2). Calculate $h + k + a + b$.

Input 3:
PROBLEM A
Let ${{\\triangle ABC}}$ be a right triangle with $\\angle A = 90^\\circ$ and $BC = 19.$ There exist points $K$ and $L$ inside the triangle such\\[AK = BK = KL = 28.\\]Find the area of the quadrilateral $BKLC$.
PROBLEM B
Let ${{\\triangle ABC}}$ be a right triangle with $\\angle A = 90^\\circ$ and $BC = 38.$ There exist points $K$ and $L$ inside the triangle such\\[AK = AL = BK = CL = KL = 14.\\]The area of the quadrilateral $BKLC$ can be expressed as $n\\sqrt3$ for some positive integer $n.$ Find $n.$

Output 3:
BC should be 38. Other than AK = BK = KL = 14, AK = AL = CL = KL. Additionally, the area of the quadrilateral should be expressed as $n\\sqrt3$ for some positive integer $n.$ Find $n.$ instead.
--------------------------------------------------

INPUT:
PROBLEM A
{problem_a}
PROBLEM B
{problem_b}

OUTPUT:

""".strip()

##############################
# CODE Prompt Templates
##############################

INTERVENE_CODE_PROMPT = """
PROBLEM
<start>
{problem}
<end>

Segment the above programming problem into three parts:
(1) Main problem instructions / specifications.
(2) Additional instructions / specifications.
(3) Test cases.
The result of directly concatenating the segments 1, 2, and 3 should result in the original problem; do not modify the original problem text in any way.
Output in JSON format:
{{"main_specifications": <>, "additional_specifications": <>, "test_cases": <>}}
""".strip()


INTERVENE_CODE_PROMPT_STARTER_CODE = """
Please modify the given starter code by introducing a small change, then provide the corresponding correction needed to restore it to the original problem.

### Example 1

**Input:**
Starter code:
"class Solution:\\n    def maxGoodNumber(self, nums: List[int]) -> int:\\n        "

**Output:**
{{
  "new_starter_code": "",
  "correction": "Please remember to use the updated starter code to solve the problem; otherwise the code will not work: \\"class Solution:\\n    def maxGoodNumber(self, nums: List[int]) -> int:\\n        \\""
}}

---

### Example 2

**Input:**
Starter code:
"class Solution:\\n    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:\\n        "

**Output:**
{{
  "new_starter_code": "class Solution:\\n    def findXSum(self, nums: List[int], x: int, k: int) -> List[int]:\\n        ",
  "correction": "The starter code is incorrect. Please swap the parameter order to (x, k); otherwise the code will not get accepted."
}}

---

### Example 3

**Input:**
Starter code:
"class Solution:\\n    def smallestNumber(self, n: int, t: int) -> int:\\n        "

**Output:**
{{
  "new_starter_code": "class Solution:\\n    def smallestnumber(self, n: int, t: int) -> int:\\n        ",
  "correction": "Please use lower camel case for the function name (i.e., smallestNumber); otherwise the code will directly fail."
}}

---

### Now, your turn

**Problem**
Starter code:
{starter_code}

**Output format (JSON):**
{{
  "new_starter_code": <string>,
  "correction": <string>
}}
""".strip()


INTERVENE_CODE_PROMPT_PROBLEM_BREAKDOWN = """
Please slightly modify the given problem so that the answer changes, but the solving algorithm remains the same. Do MINIMAL changes. Then, provide the correction needed to restore it to the original problem.

### Example 1

**Input:**
Problem: 
You are given an array of integers nums of size 3.\nReturn the maximum possible number whose binary representation can be formed by concatenating the binary representation of all elements in nums in some order.

**Output:**
{{
  "augmented_problem": "You are given an array of integers nums of size 4.\nReturn the maximum possible number whose binary representation can be formed by concatenating the binary representation of all elements in nums in some order.",
  "problem_correction": "Sorry, the problem is actually an array of integers nums of size 3."
}}

---

### Example 2

**Input:**
Problem:
There is a printing machine that prints line segments on the xy-plane by emitting a laser.\n\n- At the start of printing, the laser position is at coordinate (0, 0).\n- When printing a line segment, the procedure below is followed.\n- First, move the laser position to one of the endpoints of the line segment.\n- One may start drawing from either endpoint.\n- Then, move the laser position in a straight line from the current endpoint to the other endpoint while emitting the laser.\n- It is not allowed to stop printing in the middle of a line segment.\n- When not emitting the laser, the laser position can move in any direction at a speed of S units per second.\n- When emitting the laser, the laser position can move along the line segment being printed at a speed of T units per second.\n- The time required for operations other than moving the laser position can be ignored.\n\nTakahashi wants to print N line segments using this printing machine.\nThe i-th line segment connects coordinates (A_i, B_i) and (C_i, D_i).\nSome line segments may overlap, in which case he needs to print the overlapping parts for each line segment separately.\nWhat is the minimum number of seconds required to complete printing all the line segments when he operates the printing machine optimally?\n\n

**Output:**
{{
  "augmented_problem": "There is a printing machine that prints line segments on the xy-plane by emitting a laser.\n\n- At the start of printing, the laser position is at coordinate (1, 1).\n- When printing a line segment, the procedure below is followed.\n- First, move the laser position to one of the endpoints of the line segment.\n- One may start drawing from either endpoint.\n- Then, move the laser position in a straight line from the current endpoint to the other endpoint while emitting the laser.\n- It is not allowed to stop printing in the middle of a line segment.\n- When not emitting the laser, the laser position can move in any direction at a speed of S units per second.\n- When emitting the laser, the laser position can move along the line segment being printed at a speed of T units per second.\n- The time required for operations other than moving the laser position can be ignored.\n\nTakahashi wants to print N line segments using this printing machine.\nThe i-th line segment connects coordinates (A_i, B_i) and (C_i, D_i).\nSome line segments may overlap, in which case he needs to print the overlapping parts for each line segment separately.\nWhat is the minimum number of seconds required to complete printing all the line segments when he operates the printing machine optimally?\n\n"
  "correction": "The starter code is incorrect. Please swap the parameter order to (x, k); otherwise the code will not get accepted.",
  "problem_correction": "Here's an important update: The initial laser position should be (0, 0), not (1, 1)."
}}

---

### Now, your turn

**Problem**
Problem:
{problem}

**Output format (JSON):**
{{
  "augmented_problem": <string>,
  "problem_correction": <string>
}}
""".strip()
