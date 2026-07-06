import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model import PINN

def evaluate_and_plot(model_path="heat_2d_pinn.pt", t_eval=0.05):
    # ۱. بارگذاری مدل آموزش دیده
    model = PINN(input_dim=3, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # ۲. ساخت شبکه نقاط ارزیابی (۱۰۰ در ۱۰۰)
    x_vals = torch.linspace(0, 1, 100)
    y_vals = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    
    # زمان ارزیابی (به جای t=2 که دما صفر می‌شود، t=0.05 انتخاب شده تا دما قابل مشاهده باشد)
    T = torch.ones_like(X) * t_eval
    
    # ۳. محاسبه جواب دقیق (Exact Solution)
    X_np = X.numpy()
    Y_np = Y.numpy()
    u_exact = np.exp(-2 * (np.pi**2) * t_eval) * np.sin(np.pi * X_np) * np.sin(np.pi * Y_np)
    
    # ۴. پیش‌بینی شبکه عصبی (PINN Solution)
    with torch.no_grad():
        input_data = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
        u_pred = model(input_data).reshape(100, 100).numpy()
        
    # ۵. محاسبه خطای مطلق نقطه به نقطه
    error_absolute = np.absolute(u_exact - u_pred)
    
    # ۶. محاسبه شاخص عددی Relative L2 Error
    pinn_error_l2 = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print(f"=== Evaluation at t = {t_eval} ===")
    print(f"Relative L2 Error: {pinn_error_l2:.4e}")
    
    # ۷. رسم ویترین مقایسه‌ای (۳ نمودار در یک ردیف)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # تنظیمات مشترک محورها برای تبدیل اندیس ماتریس به مختصات هندسی دکارتی
    ticks = [0, 50, 99]
    labels = ['0.0', '0.5', '1.0']
    
    # الف) نمودار جواب دقیق
    ax0 = sns.heatmap(u_exact, cmap="jet", ax=axes[0], cbar_kws={'label': 'u(x,y)'})
    ax0.invert_yaxis()
    ax0.set_xticks(ticks); ax0.set_xticklabels(labels)
    ax0.set_yticks(ticks); ax0.set_yticklabels(labels)
    axes[0].set_title(f"Exact Solution (t={t_eval})")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    
    # ب) نمودار جواب PINN
    ax1 = sns.heatmap(u_pred, cmap="jet", ax=axes[1], cbar_kws={'label': 'u(x,y)'})
    ax1.invert_yaxis()
    ax1.set_xticks(ticks); ax1.set_xticklabels(labels)
    ax1.set_yticks(ticks); ax1.set_yticklabels(labels)
    axes[1].set_title(f"PINN Prediction (t={t_eval})")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
    
    # ج) نمودار خطای مطلق (میزان عدم انطباق)
    ax2 = sns.heatmap(error_absolute, cmap="rocket_r", ax=axes[2], cbar_kws={'label': 'Absolute Error'})
    ax2.invert_yaxis()
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels)
    ax2.set_yticks(ticks); ax2.set_yticklabels(labels)
    axes[2].set_title(f"Absolute Error (L2={pinn_error_l2:.2e})")
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")
    
    plt.tight_layout()
    plt.savefig("pinn_accuracy_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # اجرای ارزیابی در یک زمان منطقی (مثل t = 0.05)
    evaluate_and_plot("heat_2d_pinn.pt", t_eval=0.05)