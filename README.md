# Instruction Tuning GPT2 on Alpaca Dataset
- Author: Sovit Ranjan Rath _ May 6, 2024
- Practice: Mr. Jack _ May30, 2024
- Link: https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/

Fine-tuning language models to follow instructions is a major step in making them more useful. In this article, we will train the GPT2 model for following simple instructions. Instruction tuning GPT2 on the Alpaca dataset will reveal how well very small language models perform at following instructions.

In particular, we will train the GPT2 base model which contains just 124 million parameters. This is much smaller than what the industry considers as SLMs (Small Language Models), which us typically 7 bllion (7B) parameters. In fact, any language model below 3 billion parameters can be a challenge to to train for instruction following. However, in future posts, we will train many such models and see how far we can push the envelope forward. This post is a starting point for this.

~> Hôm qua (29 May 2024) tìm thấy được một chủ đề hay nên phải bắt tay vào thực hành ngay, kết quả rất tuyệt ^^ . Hồi trước mình chỉ thích thuật toán AI , không thích làm giao diện (UI/UX), không thích làm Data, ... không thích này nọ ... giờ khi làm demo AI là vướng toàn diện không chừa mục nào, từ a-z, nên va vào cái nào ngất luôn cái đó 😂 ... mình cũng đang phải ôn tập lại một số nội dung cơ bản của Javascript, UI/UX, Gradio, Dataset ... vì nếu không thông mấy cái này mà copy-paste source code của tây về thử mà nó không chạy ở một chỗ nào đó, vì một lý do nào đó thì chỉ có ngồi khóc bằng tiếng mán 😂 ... hồi trước mình nghĩ khá đơn giản là open source rất hay, chỉ cần lên github tìm cái nào hay hay thấy thích thì clone về chạy demo được thì tức là mình đã học xong phần đó ^^ không ngờ thực tế là cài đặt môi trường cũng loay hoay, chạy cũng không nổi, rồi chạy cũng lỗi không biết đâu mà lần 😂 ... vậy nên từ "nỗi đau" đó mình tự đặt ra một nguyên tắc là source code mình phải review cẩn thận, chỉnh sửa cho gọn gàng và test thử rất kỹ, đảm bảo chạy được thì mới upload lên github cho bà con thử nghiệm ^^ ... với cách làm "nghiêm túc" như vậy nên một số repo cũng có ngày đạt gần trăm lượt clone ^^ vậy mà chưa có cái repo nào được chục sao để còn nhận được huy hiệu ^^  ... đúng là "đời là bể khổ" mà 😂 <br><br>

~> Với example này, mặc dù mình đã copy y nguyên source code của tác giả nhưng ... vẫn bị lỗi :( chắc là do cài đặt môi trường và các cấu hình chương trình khác nhau bị lỗi thành ra phải fix lại một vài chỗ, sau hơn 1 ngày làm việc thì cũng đã xử lý xong ^^ <br>
~> Để thể hiện sự tôn trọng đối với tác giả, mình để nguyên phần code bản gốc,  cố gắng sửa ít nhất có thể, chỉ rào lại và thêm phần code sửa lại ở ngay dưới để vẫn đảm bảo source code chạy được và người đọc vẫn follow được phần source code nguyên bản ban đầu của tác giả ^^ <br>
~> Trong bài này tác giả dùng 'packing=True' để tự động 'concatenate different samples of similar lengths into a single batch' . Tuy nhiên khi chạy thực tế thì kết quả trả ra không chính xác, vì vậy mình đã rào chỗ đó lại để mặc định setting 'packing=False', nhưng khi đó phải viết lại preprocess_function(example), cũng phải loay hoay một lúc thì mới test xong được chỗ này ^^ . Cuối cùng thì example này cũng đã được "thuần hóa" ^^ <br>

### Tham khảo:
- https://huggingface.co/datasets/tatsu-lab/alpaca
- https://huggingface.co/docs
- https://github.com/tatsu-lab/stanford_alpaca
- https://pytorch.org/tutorials/index.html
- https://debuggercafe.com/fine-tuning-phi-1-5-using-qlora
- https://debuggercafe.com/fine-tuning-qwen-1-5-for-coding (All the training and inference shown here were carried out on a system with 10 GB RTX 3080 GPU, 10th generation i7 GPU, and 32 GB of RAM.)
- https://debuggercafe.com/spelling-correction-using-hugging-face-transformers
- https://debuggercafe.com/getting-started-with-grammar-correction
- https://debuggercafe.com/character-level-text-generation-using-lstm
- https://debuggercafe.com/word-level-text-generation-using-lstm
- https://debuggercafe.com/text-generation-with-transformers
- https://debuggercafe.com/introduction-to-gpt-1-and-gpt-2 (Question Answering: ... The smallest GPT-2 model answers less than 1% of the questions correctly even for the most common questions. On the same questions, the largest GPT-2 model answers the question correctly 5.3 times more. This shows that larger models tend to perform better on several zero-shot tasks compared to smaller models with the same architecture.)
- https://skylion007.github.io/OpenWebTextCorpus
- https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

### Screen shot:
![alt text](https://github.com/Mr-Jack-Tung/Instruction-Tuning-GPT2-on-Alpaca-Dataset/blob/main/Screenshot_Instruction-Tuning_GPT2_2024-05-30_01.jpg)

![alt text](https://github.com/Mr-Jack-Tung/Instruction-Tuning-GPT2-on-Alpaca-Dataset/blob/main/Screenshot_Instruction-Tuning_GPT2_2024-05-30_02.jpg)

### Update 31 May 2024: Sử dụng LoRA trong Instruction Tuning GPT2 ^^
```
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=16, # 16, 32, 64, 128
    lora_alpha=32, # 16, 32, 64, 128
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True,
    target_modules=[
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    ],
)

model = get_peft_model(model, peft_config)

...
trainer = SFTTrainer(
    ...
    peft_config=peft_config,
)
...
model.merge_adapter()
```
