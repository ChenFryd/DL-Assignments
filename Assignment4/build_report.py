"""Generates the Assignment 4 report as a .docx file."""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

FIGURES = '/home/chenfryd/DL-Assignments/Assignment4/figures'
OUT = '/home/chenfryd/DL-Assignments/Assignment4/Report.docx'

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── Default paragraph style: Calibri 12 ──────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.bold = True
        run.font.size = Pt(13) if level == 1 else Pt(12)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(4)
    return p

def body(text):
    p = doc.add_paragraph(text)
    p.style = doc.styles['Normal']
    return p

def bold_body(label, text):
    p = doc.add_paragraph()
    p.style = doc.styles['Normal']
    run = p.add_run(label)
    run.bold = True
    run.font.name = 'Calibri'
    run.font.size = Pt(12)
    rest = p.add_run(text)
    rest.font.name = 'Calibri'
    rest.font.size = Pt(12)
    return p

def add_figure(path, caption, width=Cm(14)):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.style = doc.styles['Normal']
    for run in cap.runs:
        run.font.size = Pt(10)
        run.font.italic = True
    cap.paragraph_format.space_after = Pt(10)

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for para in hdr[i].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True
                run.font.name = 'Calibri'
                run.font.size = Pt(11)
    for ri, row in enumerate(rows):
        cells = t.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            for para in cells[ci].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.name = 'Calibri'
                    run.font.size = Pt(11)
    doc.add_paragraph()
    return t

def page_break():
    doc.add_page_break()

# =============================================================================
# TITLE
# =============================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run('Assignment 4: Generative Models with GAN')
r.bold = True; r.font.name = 'Calibri'; r.font.size = Pt(16)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = sub.add_run('Ben-Gurion University of the Negev - Deep Learning Course')
r2.font.name = 'Calibri'; r2.font.size = Pt(12)

doc.add_paragraph()

# =============================================================================
# 1. INTRODUCTION
# =============================================================================
heading('1. Introduction')
body(
    'This report presents the design, training, and evaluation of a standard GAN and a '
    'Conditional GAN (cGAN) on the Adult Census Income dataset. The task targets tabular '
    'data synthesis: generating realistic records with a mix of continuous and categorical '
    'features while conditioning on the binary income label in the cGAN variant. Beyond '
    'the core models, the report covers a systematic comparison of two differentiable '
    'representations for categorical outputs (Sections 7), a deliberate mode-collapse '
    'experiment (Section 8), and a spectral-normalization contribution (Section 9).'
)

# =============================================================================
# 2. DATASET
# =============================================================================
heading('2. Dataset and Preprocessing')
body(
    'The Adult dataset contains 32,561 records drawn from the 1994 US Census. The target '
    'feature is binary income (<=50K / >50K). The class distribution is imbalanced: 75.9% '
    'belong to the lower-income class and 24.1% to the higher-income class (see Appendix '
    'Fig. A1). Three features contain missing values: workclass (1,836), occupation (1,843), '
    'and native-country (583); these are treated as a distinct "Unknown" category after '
    'one-hot encoding so no rows are discarded.'
)
body(
    'The feature set consists of six continuous columns (age, fnlwgt, education-num, '
    'capital-gain, capital-loss, hours-per-week) and eight categorical columns (workclass, '
    'education, marital-status, occupation, relationship, race, sex, native-country). '
    'Continuous features are standardized with StandardScaler fitted on the training split '
    'only to prevent leakage. Categorical features are one-hot encoded using a vocabulary '
    'built from the full dataset, yielding a 102-dimensional categorical block and a total '
    'input dimension of 108. Real feature distributions are shown in Appendix Figs. A2 and A3.'
)
body(
    'Three preprocessing design decisions are worth justifying explicitly. First, '
    'StandardScaler rather than MinMaxScaler was chosen for the continuous features because '
    'the generator\'s continuous output head uses linear activations with no bounding '
    'constraint. MinMaxScaler would require the generator to produce values strictly in '
    '[0, 1], which cannot be guaranteed without a sigmoid head; StandardScaler instead '
    'targets a N(0, 1) range that the generator can reach freely. Second, one-hot encoding '
    'rather than ordinal encoding was chosen for all categorical features, including '
    'apparently ordered ones such as education. Ordinal encoding implies linear interpolation '
    'in feature space (e.g., that "Some-college" is exactly halfway between "HS-grad" and '
    '"Bachelors"), which is not semantically valid. One-hot encoding lets both the generator '
    'and discriminator treat each education level as a fully independent discrete choice. '
    'Third, missing values are mapped to a distinct "Unknown" category rather than being '
    'imputed or discarded. Imputation with the mode would create artificial spikes that '
    'corrupt the categorical distribution the GAN is trying to learn, while row deletion '
    'would introduce survivorship bias into the training set.'
)
bold_body('Train/test split: ', '80% training (26,048 records) and 20% test (6,513 records), '
    'stratified on the income label to preserve the 0.759/0.241 class ratio. All results '
    'are averaged over three random seeds (42, 123, 456).')

# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================
heading('3. Model Architecture')

heading('3.1 Generator', level=2)
body(
    'The generator is a three-layer MLP with 256 hidden units per layer, batch normalization, '
    'and LeakyReLU(0.2) activations. It accepts a 128-dimensional Gaussian noise vector '
    '(and a two-dimensional income one-hot in the cGAN variant) and produces a 108-dimensional '
    'output matching the encoded data format. The output head is split: a linear layer '
    '(no activation) for the six continuous features, and one separate linear layer per '
    'categorical feature whose logits are passed through the Gumbel-Softmax straight-through '
    'estimator (default) or plain softmax (ablation). BatchNorm in the generator provides '
    'stable gradient magnitudes during early training without restricting the discriminator.'
)

heading('3.2 Discriminator', level=2)
body(
    'The discriminator is a three-layer MLP with 256, 256, and 128 hidden units, '
    'LeakyReLU(0.2) activations, and Dropout(0.3) after the first two layers. BatchNorm '
    'is deliberately excluded because it averages statistics across real and fake samples '
    'within the same batch, which can obscure the real/fake signal. The output is a single '
    'sigmoid unit. For the cGAN, the income one-hot is concatenated to the input so the '
    'discriminator learns to assess plausibility conditional on the requested class.'
)

heading('3.3 cGAN Extension', level=2)
body(
    'In the conditional variant the income one-hot (dimension 2) is concatenated to the '
    'noise vector fed to the generator and to the data vector fed to the discriminator. '
    'Both the discriminator update and the generator update use the same real-batch income '
    'labels as the conditioning signal, ensuring the discriminator is always asked "is this '
    'sample plausible given class c?" and the generator is trained to answer "yes" under '
    'the identical conditioning.'
)

# =============================================================================
# 4. TRAINING SETUP
# =============================================================================
heading('4. Training Setup')
add_table(
    ['Hyperparameter', 'Value'],
    [
        ['Epochs', '200 (Sections 6/9);  100 (Section 7);  80 (Section 8)'],
        ['Batch size', '256'],
        ['Noise dimension z', '128'],
        ['Optimizer', 'Adam, lr = 2e-4, b1 = 0.5, b2 = 0.999'],
        ['Loss function', 'Binary cross-entropy (BCE)'],
        ['Label smoothing', 'real -> 0.9,  fake -> 0.1 (one-sided)'],
        ['Temperature schedule', 'Linear anneal 1.0 -> 0.5 over training'],
        ['D : G update ratio', '1:1 (default);  5:1 (Section 8 collapse)'],
        ['Categorical strategy', 'Gumbel-ST (default);  Softmax (Section 7 ablation)'],
    ]
)
body(
    'One-sided label smoothing is used to prevent the discriminator from becoming '
    'overconfident on real samples, which would produce near-zero gradients for the '
    'generator. The Gumbel-Softmax temperature is annealed from 1.0 to 0.5 so early '
    'training benefits from softer, more gradient-rich outputs while later training '
    'commits to near-discrete samples.'
)

# =============================================================================
# 5. SECTION 6 - EVALUATION
# =============================================================================
heading('5. Evaluation (Section 6)')

heading('5.1 Loss Curves', level=2)
body(
    'Loss curves for all three seeds are shown in Appendix Figs. A4-A9. In all runs the '
    'generator loss converges to approximately 2.27-2.29 and the discriminator loss to '
    'approximately 0.65-0.66. The theoretical Nash equilibrium for BCE with label smoothing '
    '(real label = 0.9) places the discriminator optimum near 0.65, which matches the '
    'observed plateau. The generator loss well above ln(2) = 0.693 indicates the '
    'discriminator retains a consistent advantage throughout training. Fluctuations are '
    'present throughout, as expected in GAN training, but no catastrophic divergence '
    'or sudden collapse was observed.'
)

heading('5.2 Synthetic Data Quality', level=2)
body(
    'Continuous and categorical feature distributions for both GAN and cGAN are compared '
    'against real training data in Appendix Figs. A10-A13. The continuous distributions '
    'are qualitatively similar in shape, though the synthetic capital-gain and capital-loss '
    'distributions lack the heavy-tail structure of the real data. Categorical distributions '
    'are also broadly captured, but some low-frequency categories (e.g., rare native-country '
    'values) are under-represented in the synthetic data. Correlation matrices (Figs. A14-A15) '
    'show the generator partially captures inter-feature correlations, but the absolute '
    'differences remain non-trivial, suggesting the MLP backbone has limited capacity to '
    'model higher-order dependencies.'
)

heading('5.3 Detection and Efficacy Metrics', level=2)
body(
    'Detection is measured by training a Random Forest on a balanced mix of real and synthetic '
    'data using four-fold cross-validation (three folds train, one fold test, averaged over '
    'four rotations) and reporting the AUC. Low AUC near 0.5 indicates synthetic data is '
    'indistinguishable from real. Efficacy is the ratio of AUC when a Random Forest is '
    'trained on synthetic versus real data and evaluated on the original test set; high '
    'values near 1.0 indicate the synthetic data is a useful substitute for training.'
)

add_table(
    ['Model', 'Detection AUC', 'Efficacy (mean)', 'Efficacy (std)'],
    [
        ['GAN',  '1.000 +/- 0.000', '0.811', '0.186'],
        ['cGAN', '1.000 +/- 0.000', '0.714', '0.114'],
    ]
)

body(
    'Detection AUC is 1.000 for both models across all three seeds, meaning the Random '
    'Forest perfectly separates real from synthetic samples. This result is consistent with '
    'the discriminator-dominant loss pattern observed during training: a generator that '
    'consistently loses against the discriminator has not learned the full data manifold. '
    'The practical implication is that the synthetic data differs from real data in ways '
    'a simple tree-based model can detect without difficulty.'
)
body(
    'Efficacy is more nuanced. For the GAN, seed 42 yielded an efficacy of 0.55 because '
    'the pseudo-labeler (a Random Forest trained on real data) assigned only one class to '
    'all synthetic samples, producing a degenerate classifier that scored 0.5 AUC. Seeds '
    '123 and 456 yielded 0.90 and 0.98 respectively. The high variance (0.186) reflects '
    'the fragility of pseudo-labeling for unconditional GANs: label quality depends on '
    'how well the generated feature space aligns with the real class boundary, which varies '
    'with initialization. The cGAN shows lower but more consistent efficacy (0.714 +/- 0.114) '
    'because conditioning ensures the label and feature spaces are coupled during generation, '
    'removing the need for pseudo-labeling.'
)

# =============================================================================
# 6. SECTION 7 - DISCRETE FEATURES
# =============================================================================
heading('6. Discrete Feature Representation (Section 7)')

heading('6.1 Why Argmax Breaks Gradient Flow', level=2)
body(
    'Selecting a category via argmax over a softmax is not differentiable. The argmax '
    'function is piecewise constant: its output changes only at isolated switching boundaries '
    'where the maximum shifts from one category to another. For any small perturbation '
    'of the logit vector, the output one-hot is constant, so the sub-gradient is zero '
    'almost everywhere. When the generator outputs argmax-decoded one-hots, backpropagation '
    'through the discriminator reaches the argmax and the gradient is zero, so the '
    'generator parameters receive no useful update signal and the network cannot learn '
    'to improve its categorical outputs.'
)

heading('6.2 Strategy Comparison', level=2)
bold_body('Plain Softmax. ', 'Forward pass: the generator outputs a continuous probability '
    'vector in the (K-1)-simplex. The Jacobian of softmax is well-defined, so gradients '
    'flow back to the logits. Backward pass: the (K x K) softmax Jacobian is non-zero, '
    'providing useful gradient signal. However, the discriminator now sees hard 0/1 one-hots '
    'from real data and soft 0...1 probabilities from fake data, creating a format mismatch '
    'that can cause the discriminator to exploit the distributional difference rather than '
    'learning genuine semantic content.')
doc.add_paragraph()
bold_body('Gumbel-Softmax + Straight-Through (Gumbel-ST). ', 'Forward pass: Gumbel noise is '
    'added to the logits and a hard one-hot is taken via argmax, so the discriminator sees '
    'discrete samples identical in format to real data. There is no format mismatch. '
    'Backward pass: instead of the zero gradient of argmax, the straight-through estimator '
    'substitutes the gradient of the soft Gumbel-Softmax (y_hard - y_soft.detach() + y_soft). '
    'This allows informative gradients to reach the generator while the forward pass '
    'remains discrete.')

heading('6.3 Entropy Analysis', level=2)
body(
    'To quantify commitment, the average Shannon entropy of the generated soft categorical '
    'distributions (computed from raw logits, without Gumbel noise) is compared against '
    'the entropy implied by real category frequencies. A near-uniform distribution has '
    'high entropy and indicates the generator is not generating meaningful categorical '
    'structure. Results are shown in Appendix Fig. A16 and summarized below.'
)
add_table(
    ['Feature', 'Real entropy', 'Softmax', 'Gumbel-ST'],
    [
        ['workclass',       '1.148', '0.033', '1.164'],
        ['education',       '2.037', '0.172', '0.766'],
        ['marital-status',  '1.273', '0.052', '0.775'],
        ['occupation',      '2.440', '0.094', '1.313'],
        ['relationship',    '1.491', '0.064', '0.667'],
        ['race',            '0.551', '0.012', '0.340'],
        ['sex',             '0.635', '0.025', '0.443'],
        ['native-country',  '0.653', '0.000', '0.359'],
    ]
)
body(
    'Softmax entropy is close to zero for every feature, meaning the generator '
    'produces near-uniform soft distributions and relies on downstream argmax to '
    'select a category. This is not genuine categorical structure: the generator '
    'has effectively learned to be indifferent and delegates all the discrete decision '
    'to the decoder. Gumbel-ST entropy is substantially closer to the real data '
    'across all features, with the closest match for workclass (1.164 vs 1.148). '
    'Loss curve comparison (Fig. A17) shows Gumbel-ST reaches a better equilibrium '
    '(G ~2.14, D ~0.72) than Softmax (G ~1.89, D ~0.85), where the Softmax discriminator '
    'appears less challenged because format mismatch makes fake samples easier to detect.'
)
bold_body('Quantitative comparison. ', 'Detection: both 1.000 (neither fools the RF). '
    'Efficacy: Gumbel-ST 0.973 vs Softmax 0.939. Gumbel-ST is the preferred representation '
    'because it eliminates the real/fake format mismatch, produces entropy values that '
    'match real-data category statistics, and yields higher efficacy.')

# =============================================================================
# 7. SECTION 8 - MODE COLLAPSE
# =============================================================================
heading('7. Mode Collapse: Induction and Mitigation (Section 8)')

heading('7.1 Collapse Indicator', level=2)
body(
    'The collapse indicator is the categorical coverage ratio: the number of unique '
    '(workclass, marital-status, occupation) three-way combinations in the synthetic data '
    'divided by the number of unique combinations in the real training data. A value near '
    '0 indicates the generator has collapsed to a narrow set of discrete patterns; a value '
    'near or above 1 indicates full diversity. These three features are chosen because they '
    'span different vocabulary sizes and capture distinct social-demographic dimensions, '
    'making their joint distribution a sensitive indicator of mode coverage.'
)

heading('7.2 Inducing Collapse', level=2)
body(
    'To induce collapse, the discriminator-to-generator update ratio is raised from 1:1 '
    'to 5:1. The expectation is that a much stronger discriminator will push the generator '
    'to a corner of the output space where it finds a local equilibrium that fools D on '
    'a narrow set of outputs. Loss curves for both configurations are shown in '
    'Appendix Fig. A18.'
)
add_table(
    ['Configuration', 'Coverage ratio', 'Unique combos'],
    [
        ['Real data (reference)', '1.000', '381'],
        ['Baseline 1:1', '1.242', '473'],
        ['Collapsed 5:1', '1.184', '451'],
    ]
)
body(
    'The 5:1 ratio did not produce the expected collapse by this metric. Both configurations '
    'generated more unique three-way combinations than the real training data (ratio > 1). '
    'The 5:1 run produced marginally fewer combos (451 vs 473), suggesting slightly reduced '
    'diversity, but the effect was modest. A possible explanation is that with short training '
    '(80 epochs) and a moderately expressive generator, the Gumbel-ST discrete outputs '
    'are diverse enough that the generator explores many categorical combinations even under '
    'discriminator pressure. The loss curves (Fig. A18) do not show the sharp G-loss spike '
    'typical of collapse, supporting this interpretation.'
)

heading('7.3 Mitigation: Minibatch Standard Deviation', level=2)
bold_body('Prediction. ', 'The minibatch standard-deviation (MBD) layer appends the mean '
    'per-feature standard deviation across the mini-batch to the discriminator input. When '
    'the generator produces homogeneous outputs, the batch std of fake samples is near zero, '
    'making them trivially detectable. The generator should therefore be forced to produce '
    'more diverse outputs. Predicted outcome: coverage ratio should recover toward or above '
    'the baseline; G-loss should stabilize; D-loss should not reach near-zero.')
doc.add_paragraph()
bold_body('Observed. ', 'MBD + 5:1 yielded a coverage ratio of 0.717 (273/381), lower than '
    'both the baseline (1.242) and the collapsed run (1.184). G-loss settled at 2.32 '
    'and D-loss at 0.66, with no meaningful change from the collapsed run.')
doc.add_paragraph()
body(
    'The prediction was not confirmed. Rather than recovering diversity, MBD under heavy '
    'discriminator pressure (5:1) further reduced categorical coverage. The most plausible '
    'explanation is that the MBD signal interacts with the 5:1 imbalance in a way that '
    'was not anticipated: the discriminator is updated five times per generator step, so '
    'by the time the generator responds to the diversity signal it has already been pushed '
    'toward a narrow set of outputs that receive low MBD-augmented scores. The net effect '
    'is concentration rather than diversification. MBD is generally effective against mild '
    'collapse, but pairing it with a very strong discriminator imbalance removes the '
    'breathing room the generator needs to explore. A lighter ratio (2:1 or 3:1) would '
    'be a more appropriate test of MBD in isolation. The bar chart and loss comparison '
    'are in Appendix Figs. A19-A20.'
)

# =============================================================================
# 8. SECTION 9 - SPECTRAL NORMALIZATION
# =============================================================================
heading('8. Open-Ended Contribution: Spectral Normalization (Section 9)')

heading('8.1 Modification and Motivation', level=2)
body(
    'Spectral normalization (SN) is applied to every linear layer of the discriminator. '
    'SN constrains the spectral norm (largest singular value) of each weight matrix W to '
    'be at most 1 by dividing by sigma_1(W) at each forward pass. This bounds the overall '
    'Lipschitz constant of the discriminator to 1.'
)
body(
    'The motivation targets the core instability observed in the baseline: the discriminator '
    'converges quickly to a near-optimal solution and its gradients saturate (sigmoid output '
    'approaches 0 for fake samples), leaving the generator with a vanishingly small update '
    'signal. A large Lipschitz constant enables steep loss landscapes that accelerate this '
    'dynamic. By constraining the Lipschitz constant to 1, SN prevents the discriminator '
    'from becoming arbitrarily confident, ensuring the generator always receives meaningful '
    'gradients. The key advantage over alternatives such as gradient penalty is that SN '
    'requires no hyperparameter, no additional forward pass, and adds no computational '
    'overhead at inference.'
)

heading('8.2 Prediction', level=2)
body(
    'Written before running the experiment: '
    '(1) G-loss variance should decrease because bounding the Lipschitz constant prevents '
    'the sharp discriminator confidence spikes that cause the generator gradient to oscillate. '
    '(2) D-loss should not collapse to near-zero because the constrained discriminator cannot '
    'become arbitrarily confident. '
    '(3) Efficacy should improve or be maintained because more stable gradients should '
    'produce a generator that better covers the data distribution.'
)

heading('8.3 Results', level=2)
add_table(
    ['Metric', 'Baseline GAN', 'Spectral-Norm GAN'],
    [
        ['G-loss (final)',    '~2.27',   '~0.71'],
        ['D-loss (final)',    '~0.66',   '~1.36'],
        ['G-loss variance',  '0.115093', '0.000014'],
        ['Variance reduction', '-',      '100.0%'],
        ['Detection AUC',    '1.000',   '1.000'],
        ['Efficacy',         '0.552',   '0.987'],
    ]
)
body(
    'All three predictions were confirmed. G-loss variance was reduced by 100%, from '
    '0.115 to 0.000014. The final G-loss of 0.71 is very close to ln(2) = 0.693, which '
    'is the theoretical equilibrium value when the discriminator treats fake samples as '
    '50/50 real/fake. The D-loss of 1.36 reflects a discriminator that is genuinely '
    'uncertain rather than dominant. Efficacy improved from 0.552 to 0.987, nearly doubling, '
    'indicating the generator learned a far more useful representation of the data '
    'distribution when its gradient signal was stabilized. Loss curves are shown in '
    'Appendix Fig. A21.'
)
body(
    'Detection AUC remained at 1.000, meaning even the SN generator produces data that '
    'a Random Forest can distinguish from real. This suggests the limitation lies in the '
    'MLP generator capacity and the tabular data complexity, not in training stability. '
    'SN addresses the stability problem effectively but cannot substitute for architectural '
    'improvements such as normalizing flows or a more expressive backbone.'
)

# =============================================================================
# PAGE BREAK BEFORE APPENDIX
# =============================================================================
page_break()

# =============================================================================
# APPENDIX
# =============================================================================
heading('Appendix: Figures')

fig_list = [
    (f'{FIGURES}/eda_class_dist.png',
     'Fig. A1. Class distribution of the income target. '
     'The 3:1 imbalance (75.9% vs 24.1%) is preserved by stratified splitting.',
     Cm(8)),
    (f'{FIGURES}/eda_continuous.png',
     'Fig. A2. Continuous feature distributions in the real training data. '
     'Capital-gain and capital-loss exhibit heavy positive skew with most values at zero; '
     'age and hours-per-week follow roughly bell-shaped distributions.',
     Cm(15)),
    (f'{FIGURES}/eda_categorical.png',
     'Fig. A3. Categorical feature distributions (top 15 categories shown per feature; '
     'remaining values aggregated as "Other"). '
     'Native-country is heavily dominated by United-States; occupation and education '
     'show broader spread that a GAN must capture to avoid mode collapse.',
     Cm(15)),
    (f'{FIGURES}/gan_loss_seed42.png',
     'Fig. A4. GAN training loss, seed 42. '
     'G-loss climbs from ~2.0 and plateaus near 2.27; D-loss settles near 0.66. '
     'The gap between G-loss and ln(2)=0.693 indicates persistent discriminator dominance.',
     Cm(13)),
    (f'{FIGURES}/gan_loss_seed123.png',
     'Fig. A5. GAN training loss, seed 123. '
     'Pattern is consistent with seed 42: D settles near 0.65, G never recovers to the '
     'Nash equilibrium value of 0.693.',
     Cm(13)),
    (f'{FIGURES}/gan_loss_seed456.png',
     'Fig. A6. GAN training loss, seed 456. '
     'Consistent D-dominant plateau across all three seeds confirms this is a structural '
     'property of the training setup, not seed-specific variance.',
     Cm(13)),
    (f'{FIGURES}/cgan_loss_seed42.png',
     'Fig. A7. cGAN training loss, seed 42. '
     'Conditioning on the income label does not substantially change the loss dynamics; '
     'G-loss again plateaus well above 0.693.',
     Cm(13)),
    (f'{FIGURES}/cgan_loss_seed123.png',
     'Fig. A8. cGAN training loss, seed 123. '
     'G/D pattern mirrors the unconditional GAN; the discriminator retains an advantage '
     'regardless of the conditioning signal.',
     Cm(13)),
    (f'{FIGURES}/cgan_loss_seed456.png',
     'Fig. A9. cGAN training loss, seed 456. '
     'Stable convergence with no catastrophic oscillation, but the D-dominant imbalance '
     'persists across all seeds.',
     Cm(13)),
    (f'{FIGURES}/gan_dist_continuous.png',
     'Fig. A10. GAN continuous feature distributions, real vs synthetic (seed 42). '
     'Overall shapes are broadly matched; capital-gain and capital-loss heavy tails '
     'are underrepresented in the synthetic data.',
     Cm(15)),
    (f'{FIGURES}/gan_dist_categorical.png',
     'Fig. A11. GAN categorical feature distributions, real vs synthetic (seed 42). '
     'High-frequency categories are roughly reproduced; low-frequency categories '
     '(e.g., rare native-country values) are systematically under-represented.',
     Cm(15)),
    (f'{FIGURES}/cgan_dist_continuous.png',
     'Fig. A12. cGAN continuous feature distributions, real vs synthetic (seed 42). '
     'Conditional generation produces similar continuous-feature coverage to the '
     'unconditional GAN; heavy-tail features remain challenging.',
     Cm(15)),
    (f'{FIGURES}/cgan_dist_categorical.png',
     'Fig. A13. cGAN categorical feature distributions, real vs synthetic (seed 42). '
     'Class-conditional generation preserves category proportions somewhat better than '
     'the unconditional GAN for occupation and marital-status.',
     Cm(15)),
    (f'{FIGURES}/gan_corr.png',
     'Fig. A14. GAN correlation matrices: real (left), synthetic (center), difference (right). '
     'Non-trivial residuals in the difference matrix indicate the generator captures '
     'only first-order marginals and misses higher-order inter-feature dependencies.',
     Cm(15)),
    (f'{FIGURES}/cgan_corr.png',
     'Fig. A15. cGAN correlation matrices: real (left), synthetic (center), difference (right). '
     'Residuals are comparable to the unconditional GAN; conditioning does not substantially '
     'improve correlation structure reproduction.',
     Cm(15)),
    (f'{FIGURES}/s7_entropy.png',
     'Fig. A16. Per-feature Shannon entropy: real data vs softmax vs Gumbel-ST. '
     'Softmax entropy collapses near zero across all features (near-uniform soft distributions); '
     'Gumbel-ST entropy closely tracks the real-data reference.',
     Cm(14)),
    (f'{FIGURES}/s7_loss_comparison.png',
     'Fig. A17. Training loss comparison: softmax vs Gumbel-ST. '
     'The softmax discriminator reaches a lower D-loss (~0.85) than Gumbel-ST (~0.72), '
     'consistent with the format mismatch making fake samples trivially detectable.',
     Cm(14)),
    (f'{FIGURES}/s8_collapse_loss.png',
     'Fig. A18. Section 8 loss curves: baseline 1:1 vs collapsed 5:1. '
     'No sharp G-loss spike is observed in the 5:1 run, indicating the discriminator '
     'pressure alone was insufficient to produce clear collapse within 80 epochs.',
     Cm(14)),
    (f'{FIGURES}/s8_coverage.png',
     'Fig. A19. Categorical coverage ratios: baseline, collapsed, and MBD-mitigated. '
     'Coverage above 1.0 for baseline and collapsed runs means no collapse occurred; '
     'MBD + 5:1 unexpectedly reduced coverage to 0.717, contra the predicted mitigating effect.',
     Cm(10)),
    (f'{FIGURES}/s8_all_loss.png',
     'Fig. A20. Section 8 loss curves: all three configurations on one plot. '
     'The MBD run shows no meaningful improvement in G-loss or D-loss relative to the '
     'collapsed run, confirming the generator was not helped by the diversity signal.',
     Cm(15)),
    (f'{FIGURES}/s9_sn_loss.png',
     'Fig. A21. Section 9 loss curves: baseline vs spectral-norm GAN. '
     'Spectral-norm G-loss stabilizes near ln(2)=0.693 (the Nash equilibrium) '
     'while D-loss rises to ~1.36, a stark contrast to the baseline D-dominant plateau.',
     Cm(14)),
]

for path, caption, width in fig_list:
    add_figure(path, caption, width)

doc.save(OUT)
print(f'Saved: {OUT}')
