import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, r2_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

def print_metrics(val, pred):
    criteria = ['Background', 'Lighting', 'Focus', 'Orientation', 'Color calibration', 'Resolution', 'Field of view']
    print(f"\n{'Criteria':^18} | {'MAE':^10} | {'R^2':^10} | {'SRCC':^10} | {'Cohens Kappa':^14} |")
    print("----------------------------------------------------------------------------")

    for i in range(pred.shape[1]):
        mae_value = mean_absolute_error(val[:, i], pred[:, i])
        r2_value = r2_score(val[:, i], pred[:, i])
        spearman_corr, _ = stats.spearmanr(val[:, i], pred[:, i]) if np.std(val[:, i]) > 0 and np.std(pred[:, i]) > 0 else (np.nan, np.nan)
        kappa = cohen_kappa_score(val[:, i].astype(int), pred[:, i].astype(int), weights='quadratic')  # Assuming ordinal scale
        
        print(f"{criteria[i]:^18} | {mae_value:^10.4f} | {r2_value:^10.4f} | {spearman_corr:^10.4f} | {kappa:^14.4f} |")

    global_metrics = {
        'Global MAE': mean_absolute_error(val.flatten(), pred.flatten()),
        'Global R^2': r2_score(val.flatten(), pred.flatten()),
        'Global SRCC': stats.spearmanr(val.flatten(), pred.flatten())[0] if np.std(val.flatten()) > 0 and np.std(pred.flatten()) > 0 else np.nan,
        'Global Cohens Kappa': cohen_kappa_score(val.flatten().astype(int), pred.flatten().astype(int), weights='quadratic')
    }

    print(f"\n{'MAE':^10} | {'R^2':^10} | {'SRCC':^10} | {'Cohens Kappa':^14} |")
    print("-------------------------------------------------------")
    print(f"{global_metrics['Global MAE']:^10.4f} | {global_metrics['Global R^2']:^10.4f} | {global_metrics['Global SRCC']:^10.4f} | {global_metrics['Global Cohens Kappa']:^14.4f} |")


def plot_all_confusion_matrices(y_true, y_pred):
    criteria = ['Background', 'Lighting', 'Focus', 'Orientation', 'Color calibration', 'Resolution', 'Field of view']
    fig, axes = plt.subplots(1, 7, figsize=(35, 5), sharey=True)
    for i, ax in enumerate(axes.flatten()):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(criteria[i], fontsize=12)
        ax.set_xlabel('Predicted Scores')
        ax.set_ylabel('Actual Scores')
        # Set ticks and labels explicitly if needed
        ticks = np.arange(cm.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
    plt.tight_layout()
    plt.show()

def plot_prediction_scores(val_scores, predictions):
    distortion_criteria = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    fig, axes = plt.subplots(1, len(distortion_criteria), figsize=(20, 4), sharey=True, sharex=True)
    
    for i, ax in enumerate(axes):
        # Prepare data for mean and std deviation
        unique_actuals = np.unique(val_scores[:, i])
        mean_predictions = [np.mean(predictions[val_scores[:, i] == x]) for x in unique_actuals]
        std_predictions = [np.std(predictions[val_scores[:, i] == x]) for x in unique_actuals]

        # Mean prediction line
        ax.plot(unique_actuals, mean_predictions, 'b-', label='Mean Predictions')
        
        # Standard deviation shading
        ax.fill_between(unique_actuals, 
                        np.array(mean_predictions) - np.array(std_predictions),
                        np.array(mean_predictions) + np.array(std_predictions), 
                        color='blue', alpha=0.2, label='±1 Std Dev')

        # Diagonal line for perfect predictions
        ax.plot([val_scores.min(), val_scores.max()], [val_scores.min(), val_scores.max()], 'r--', lw=2, label='Perfect Fit')
        ax.set_xlim(val_scores.min(), val_scores.max())
        ax.set_ylim(val_scores.min(), val_scores.max())
        ax.set_title(distortion_criteria[i])
        ax.set_xlabel('Actual Scores')
        ax.grid(True)

    axes[0].set_ylabel('Predicted Scores')
    plt.tight_layout()

    # Central legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize='small')

    plt.show()


def plot_results1(original_images, distorted_images, predictions, actuals, num_plots=9):
    criteria_names = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    num_criteria = len(criteria_names)
    angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, axs = plt.subplots(num_plots, 4, figsize=(20, num_plots * 5))  # 4 columns: Original Image, Distorted Image, Actual Radar, Prediction Radar

    for i in range(num_plots):
        # Original Image
        axs[i, 0].imshow(original_images[i])
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        # Distorted Image
        axs[i, 1].imshow(distorted_images[i])
        axs[i, 1].set_title('Distorted Image')
        axs[i, 1].axis('off')

        # Actual Radar chart
        ax_actual = plt.subplot(num_plots, 4, 4 * i + 3, polar=True)
        ax_actual.set_theta_offset(np.pi / 2)
        ax_actual.set_theta_direction(-1)
        plt.xticks(angles[:-1], criteria_names)

        ax_actual.set_ylim(0, 1)
        ax_actual.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_actual.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

        actual_values = actuals[i].tolist()
        actual_values += actual_values[:1]
        ax_actual.plot(angles, actual_values, linewidth=2, linestyle='solid', label='Actual', color='red')
        ax_actual.fill(angles, actual_values, 'r', alpha=0.1)
        plt.title(f'Actual Radar Chart {i+1}', size=12, y=1.1)

        # Prediction Radar chart
        ax_pred = plt.subplot(num_plots, 4, 4 * i + 4, polar=True)
        ax_pred.set_theta_offset(np.pi / 2)
        ax_pred.set_theta_direction(-1)
        plt.xticks(angles[:-1], criteria_names)

        ax_pred.set_ylim(0, 1)
        ax_pred.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_pred.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

        pred_values = predictions[i].tolist()
        pred_values += pred_values[:1]
        ax_pred.plot(angles, pred_values, linewidth=2, linestyle='solid', label='Prediction', color='blue')
        ax_pred.fill(angles, pred_values, 'b', alpha=0.1)
        plt.title(f'Prediction Radar Chart {i+1}', size=12, y=1.1)

    plt.tight_layout()
    plt.show()

def plot_results(original_images, distorted_images, predictions, actuals, num_plots=9):
    criteria_names = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    num_criteria = len(criteria_names)
    angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    if distorted_images is not None:
        fig, axs = plt.subplots(num_plots, 4, figsize=(20, num_plots * 5))  # 4 columns: Original Image, Distorted Image, Actual Radar, Prediction Radar
    else:
        fig, axs = plt.subplots(num_plots, 3, figsize=(15, num_plots * 5))  # 3 columns: Original Image, Actual Radar, Prediction Radar

    for i in range(num_plots):
        # Original Image
        axs[i, 0].imshow(original_images[i])
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        if distorted_images is not None:
            # Distorted Image
            axs[i, 1].imshow(distorted_images[i])
            axs[i, 1].set_title('Distorted Image')
            axs[i, 1].axis('off')

            # Actual Radar chart
            ax_actual = plt.subplot(num_plots, 4, 4 * i + 3, polar=True)
        else:
            # Actual Radar chart
            ax_actual = plt.subplot(num_plots, 3, 3 * i + 2, polar=True)

        ax_actual.set_theta_offset(np.pi / 2)
        ax_actual.set_theta_direction(-1)
        plt.xticks(angles[:-1], criteria_names)

        ax_actual.set_ylim(0, 1)
        ax_actual.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_actual.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

        actual_values = actuals[i].tolist()
        actual_values += actual_values[:1]
        ax_actual.plot(angles, actual_values, linewidth=2, linestyle='solid', label='Actual', color='red')
        ax_actual.fill(angles, actual_values, 'r', alpha=0.1)
        plt.title(f'Actual Radar Chart {i+1}', size=12, y=1.1)

        if distorted_images is not None:
            # Prediction Radar chart
            ax_pred = plt.subplot(num_plots, 4, 4 * i + 4, polar=True)
        else:
            # Prediction Radar chart
            ax_pred = plt.subplot(num_plots, 3, 3 * i + 3, polar=True)

        ax_pred.set_theta_offset(np.pi / 2)
        ax_pred.set_theta_direction(-1)
        plt.xticks(angles[:-1], criteria_names)

        ax_pred.set_ylim(0, 1)
        ax_pred.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_pred.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

        pred_values = predictions[i].tolist()
        pred_values += pred_values[:1]
        ax_pred.plot(angles, pred_values, linewidth=2, linestyle='solid', label='Prediction', color='blue')
        ax_pred.fill(angles, pred_values, 'b', alpha=0.1)
        plt.title(f'Prediction Radar Chart {i+1}', size=12, y=1.1)

    plt.tight_layout()
    plt.show()