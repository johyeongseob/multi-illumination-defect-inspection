from PIL import Image, ImageDraw, ImageFont

# 이미지 크기 설정
image_width = 100
image_height = 100

# # Linux
# font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# font_size = 30
# font = ImageFont.truetype(font_path, font_size)

# Window
font_path = "C:\\Windows\\Fonts\\arial.ttf"
font_size = 35
font = ImageFont.truetype(font_path, font_size)

# 이미지를 담을 리스트 생성
images = []

# 숫자 리스트
numbers = ["1", "2", "3", "4", "5", "8", "11"]

# 숫자마다 이미지 생성
for number in numbers:
    # 이미지 생성
    image = Image.new("RGB", (image_width, image_height), "black")
    draw = ImageDraw.Draw(image)

    # 텍스트 작성
    text_width, text_height = draw.textsize(number, font=font)
    x = (image_width - text_width) // 2
    y = (image_height - font_size) // 2
    draw.text((x, y), number, fill="white", font=font)

    # 리스트에 이미지 추가
    images.append(image)

# 이미지를 가로로 붙이기
combined_image = Image.new("RGB", (image_width * len(numbers), image_height), "black")
x_offset = 0
for image in images:
    combined_image.paste(image, (x_offset, 0))
    x_offset += image_width

# 이미지 저장 또는 표시
# combined_image.show()
combined_image.save("combined_numbers.png")  # 이미지 저장 시 주석 해제

result_image =  Image.new("RGB", (image_width * len(numbers) * 2 + 2, image_height), "black")
x_offset = 0
for image in range(2):
    result_image.paste(combined_image, (x_offset, 0))
    x_offset += 700 + 2

result_image.save("result_image.png")  # 이미지 저장 시 주석 해제

