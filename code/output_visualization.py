import os
from PIL import Image, ImageDraw, ImageFont


class ImageResultProcessor:
    def __init__(self, preds, targets, probabilities):
        self.preds = preds
        self.targets = targets
        self.img_cases = [(index, pred, label) for index, (pred, label) in enumerate(zip(preds, targets))]
        self.probabilities = probabilities

    def false_cases(self):
        false_cases = [(index, pred, label) for index, (pred, label) in enumerate(zip(self.preds, self.targets)) if pred != label]
        return false_cases

    def concat_images(self, images):
        # 이미지들을 가로로 연결한 이미지 생성
        total_width = sum(img.width for img in images) + len(images)
        max_height = max(img.height for img in images)
        new_image = Image.new('RGB', (total_width, max_height + 35), color='black')

        x_offset = 0
        draw = ImageDraw.Draw(new_image)
        for idx, img in enumerate(images):
                new_image.paste(img, (x_offset, 35))
                x_offset += img.width
                if idx != len(images) - 1:
                    draw.line([(x_offset, 35), (x_offset, max_height + 35)], fill='white', width=1)  # Add boundary line except last one
                    x_offset += 1

        return new_image

    def save_images_with_info(self, test_indices, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        class_map = {0: 'NG1', 1: 'NG2', 2: 'NG3', 3: 'OK'}

        views_images = []
        for key, value in test_indices.items():
            images = [Image.open(img_path) for img_path in value[0]]
            views_image = self.concat_images(images)

            for order in range(len(self.img_cases)):
                if self.img_cases[order][0] == key:
                    num, pred, label = self.img_cases[order][0], self.img_cases[order][1], self.img_cases[order][2]
            info = f"Num: {(key % 12) +1}, Pred: {class_map[pred]}, Label: {class_map[label]}, Softmax: [" \
                   f"{self.probabilities[key][0]:.2f}, " \
                   f"{self.probabilities[key][1]:.2f}, " \
                   f"{self.probabilities[key][2]:.2f}, " \
                   f"{self.probabilities[key][3]:.2f} ]"

            # 이미지에 텍스트 추가
            draw = ImageDraw.Draw(views_image)
            font = ImageFont.truetype("arial.ttf", 20)
            print(key)
            if key < 3:
                text_color = (0, 255, 0) if pred != label and pred in [0, 1, 2] else (255, 0, 0) if pred != label and pred == 3 else (255, 255, 255)
            else:
                text_color = (255, 0, 0) if pred != label else (255, 255, 255) # Red legend for false image
            draw.text((10, 10), info, fill=text_color, font=font)

            views_images.append(views_image)

        print(len(views_images))

        result_images = []
        for i in range(len(views_images) // 12):
            img = views_images[0]
            new_image = Image.new('RGB', (2 * img.width + 1, 6 * img.height + 5))
            for y in range(6):
                for x in range(2):
                    index = i * 12 + y * 2 + x
                    if index < len(views_images):
                        new_image.paste(views_images[index], (x * img.width, y * img.height))
            draw = ImageDraw.Draw(new_image)
            draw.line([(img.width, 0), (img.width, img.height * 6)], fill=(255, 255, 255), width=2)

            result_images.append(new_image)

        print(len(result_images))

        num_path = 'result_image.png'
        num_img = Image.open(num_path)
        for idx, result in enumerate(result_images):
            new_image = Image.new('RGB', (result.width, result.height + num_img.height))
            new_image.paste(num_img, (0,0))
            new_image.paste(result, (0, num_img.height))

            # 결과 이미지 저장
            output_path = os.path.join(output_folder, f"result_{idx+1}.png")
            new_image.save(output_path)


# e.g. Generate instant

# if __name__ == '__main__':
#     preds와 targets를 입력으로 받아 ImageResultProcessor 객체 생성
#     processor = ImageResultProcessor(preds, targets)
#
#     False cases count 및 False cases 리스트 얻기
#     false_cases = processor.count_false_cases()
#
#     결과 이미지 저장하기
#     output_folder = "result_images"
#     test1_indices = {...}  # test1_indices를 딕셔너리로 대체
#     processor.save_images_with_info(test1_indices, output_folder)