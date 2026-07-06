import yaml
import torch
import torch.optim as optim
from model import PINN
from pinn import calculate_loss
from plot import evaluate_and_plot

def main():
    # ۱. بارگذاری فایل تنظیمات
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # ۲. مقداردهی اولیه مدل با توجه به تنظیمات فایل YAML
    # استفاده از متد get برای امنیت بیشتر و جلوگیری از خطای KeyError
    net_config = config.get('network', {})
    model = PINN(
        input_dim=net_config.get('input_dim', 3),
        hidden_dim=net_config.get('hidden_dim', 64),
        output_dim=net_config.get('output_dim', 1)
    )
    
    # ۳. تنظیم بهینه‌ساز (Optimizer)
    lr = float(config['training']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    iterations = config['training']['iterations']
    print_every = config['training']['print_every']

    # ۴. تلاش برای بارگذاری وزن‌های قبلی (ادامه آموزش روی بهترین مدل قبلی)
    try:
        model.load_state_dict(torch.load("heat_2d_pinn.pt"))
        print("Previous weights loaded successfully! Continuing training...")
    except FileNotFoundError:
        print("No previous weights found. Starting training from scratch.")
    
    # ۵. تعریف متغیرهای لازم برای ذخیره هوشمند و شرطی مدل
    model_path = "heat_2d_pinn.pt"
    best_loss = float('inf')  # شروع مقایسه از لوس بی‌نهایت
    
    print("\nStarting 2D Heat Equation PINN Training...")
    
    # ۶. حلقه اصلی آموزش شبکه
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # محاسبه مقدار لوس فیزیک و شرایط مرزی/اولیه
        loss = calculate_loss(model, config)
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        # ۷. شرط طلایی: وزن‌ها فقط و فقط زمانی ذخیره می‌شوند که لوس فعلی از تمام تکرارهای قبلی کمتر باشد
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), model_path)  # ذخیره آنی بهترین وزن‌ها
        
        # نمایش وضعیت آموزش در فواصل مشخص
        if iteration % print_every == 0:
            print(f"Iteration {iteration:4d} | Current Loss: {current_loss:.4e} | Best Loss: {best_loss:.4e}")
            
    print(f"\nTraining finished! Best achieved loss was: {best_loss:.4e}")

    # ۸. رسم خودکار و اعتبارسنجی مقایسه‌ای با حل دقیق پس از پایان دوره آموزش
    print("Generating 2D Heat Equation plots...")
    evaluate_and_plot(model_path) 

if __name__ == "__main__":
    main()