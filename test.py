def classify_words(input_string):
    words = input_string.split()
    classified_words = []

    for word in words:
        if word.isdigit():
            classified_words.append("NUMBER")
        elif word.isalpha():
            classified_words.append("WORD")
        else:
            classified_words.append("MIXED")

    return " ".join(classified_words)

# 主函数
def main():
    try:
        while True:
            input_string = input()
            if input_string:
                output_string = classify_words(input_string)
                print(output_string)
            else:
                break
    except EOFError:
        pass

if __name__ == "__main__":
    main()

