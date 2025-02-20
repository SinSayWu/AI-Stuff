path = "C:\Users\yuhan\OneDrive\github\yuhangwu2009@gmail.com\python\AI Coding\TextGeneration\TextGenerationBooks\HP Books\ASC.txt"
story = ""
with open(path) as f:
    for _ in range(len(f)):
        line = next(f)
        print(line)
        story.append(line)

print(len(story))