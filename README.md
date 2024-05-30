# Instruction Tuning GPT2 on Alpaca Dataset
- Author: Sovit Ranjan Rath _ May 6, 2024
- Practice: Mr. Jack _ May30, 2024
- Link: https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/

Fine-tuning language models to follow instructions is a major step in making them more useful. In this article, we will train the GPT2 model for following simple instructions. Instruction tuning GPT2 on the Alpaca dataset will reveal how well very small language models perform at following instructions.

In particular, we will train the GPT2 base model which contains just 124 million parameters. This is much smaller than what the industry considers as SLMs (Small Language Models), which us typically 7 bllion (7B) parameters. In fact, any language model below 3 billion parameters can be a challenge to to train for instruction following. However, in future posts, we will train many such models and see how far we can push the envelope forward. This post is a starting point for this.

~> Hôm qua (29 May 2024) tìm thấy được một chủ đề hay nên phải bắt tay vào thực hành ngay, kết quả rất tuyệt ^^ <br>
~> Mặc dù copy y nguyên source code của tác giả nhưng ... bị lỗi :( thành ra phải fix lại một vài chỗ, sau hơn 1 ngày làm việc thì cũng đã xử lý xong ^^ <br>
~> Để thể hiện sự tôn trọng đối với tác giả, mình để nguyên phần code bản gốc, chỉ rào lại và thêm phần code sửa lại ở ngay dưới để vẫn đảm bảo source code chạy được và người đọc vẫn follow được phần source code nguyên bản ban đầu của tác giả ^^
