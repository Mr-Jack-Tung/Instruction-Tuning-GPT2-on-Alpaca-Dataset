# Instruction Tuning GPT2 on Alpaca Dataset
- Author: Sovit Ranjan Rath _ May 6, 2024
- Practice: Mr. Jack _ May30, 2024
- Link: https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/

Fine-tuning language models to follow instructions is a major step in making them more useful. In this article, we will train the GPT2 model for following simple instructions. Instruction tuning GPT2 on the Alpaca dataset will reveal how well very small language models perform at following instructions.

In particular, we will train the GPT2 base model which contains just 124 million parameters. This is much smaller than what the industry considers as SLMs (Small Language Models), which us typically 7 bllion (7B) parameters. In fact, any language model below 3 billion parameters can be a challenge to to train for instruction following. However, in future posts, we will train many such models and see how far we can push the envelope forward. This post is a starting point for this.

~> Hôm qua (29 May 2024) tìm thấy được một chủ đề hay nên phải bắt tay vào thực hành ngay, kết quả rất tuyệt ^^ . Hồi trước mình chỉ thích thuật toán AI , không thích làm giao diện (UI/UX), không thích làm Data, ... không thích này nọ ... giờ khi làm demo AI là vướng toàn diện không chừa mục nào, từ a-z, nên va vào cái nào ngất luôn cái đó 😂 ... mình cũng đang phải ôn tập lại một số nội dung cơ bản của Javascript, UI/UX, Gradio, Dataset ... vì nếu không thông mấy cái này mà copy-paste source code của tây về thử mà nó không chạy ở một chỗ nào đó, vì một lý do nào đó thì chỉ có ngồi khóc bằng tiếng mán 😂 ... hồi trước mình nghĩ khá đơn giản là open source rất hay, chỉ cần lên github tìm cái nào hay hay thấy thích thì clone về chạy demo được thì tức là mình đã học xong phần đó ^^ không ngờ thực tế là cài đặt môi trường cũng loay hoay, chạy cũng không nổi, rồi chạy cũng lỗi không biết đâu mà lần 😂 ... vậy nên từ "nỗi đau" đó mình tự đặt ra một nguyên tắc là source code mình phải review cẩn thận, chỉnh sửa cho gọn gàng và test thử rất kỹ, đảm bảo chạy được thì mới upload lên github cho bà con thử nghiệm ^^ ... với cách làm "nghiêm túc" như vậy nên một số repo cũng có ngày đạt gần trăm lượt clone ^^ vậy mà chưa có cái repo nào được chục sao để còn nhận được huy hiệu ^^  ... đúng là "đời là bể khổ" mà 😂 <br><br>

~> Với example này, mặc dù mình đã copy y nguyên source code của tác giả nhưng ... vẫn bị lỗi :( lúc đầu thì cũng chưa tin là source code của tác giả bị lỗi đâu, nhưng thử nghiệm vài lần vẫn bị lỗi nên thành ra phải fix lại một vài chỗ, sau hơn 1 ngày làm việc thì cũng đã xử lý xong ^^ <br>
~> Để thể hiện sự tôn trọng đối với tác giả, mình để nguyên phần code bản gốc,  cố gắng sửa ít nhất có thể, chỉ rào lại và thêm phần code sửa lại ở ngay dưới để vẫn đảm bảo source code chạy được và người đọc vẫn follow được phần source code nguyên bản ban đầu của tác giả ^^ <br>

### Tham khảo:
- https://huggingface.co/datasets/tatsu-lab/alpaca
- https://huggingface.co/docs
- https://github.com/tatsu-lab/stanford_alpaca
- https://pytorch.org/tutorials/index.html

### Screen shot:
![alt text](https://github.com/Mr-Jack-Tung/Instruction-Tuning-GPT2-on-Alpaca-Dataset/blob/main/Screenshot_Instruction-Tuning_GPT2_2024-05-30_01.jpg)

![alt text](https://github.com/Mr-Jack-Tung/Instruction-Tuning-GPT2-on-Alpaca-Dataset/blob/main/Screenshot_Instruction-Tuning_GPT2_2024-05-30_02.jpg)
