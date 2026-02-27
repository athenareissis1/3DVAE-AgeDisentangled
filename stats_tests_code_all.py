   ##### STATISTICAL TESTS #####

def compute_mi(self, latents, factors):
    """
    Compute Mutual Information Gap (MIG) - measures the gap between the top two mutual information scores.
    latents: (N, D) latent representations (e.g., identity latents)
    factors: (N, K) ground truth factors (e.g., age latents)
    """
    
    mi_matrix = np.zeros((latents.shape[1], factors.shape[1]))

    for i in range(latents.shape[1]): 
        for j in range(factors.shape[1]):  
            mi_matrix[i, j] = mutual_info_regression(latents[:, i].reshape(-1, 1), factors[:, j]).mean()

    mi = np.mean(mi_matrix)
    
    return mi
    
def compute_dci(self, latents, factors):
    """
    Compute DCI (Disentanglement, Completeness, Informativeness) for the entire latent representations.
    latents: (N, D1) latent representations (e.g., identity latents)
    factors: (N, D2) ground truth factors (e.g., age latents)
    Returns:
        disentanglement: Disentanglement score
        completeness: Completeness score
        informativeness: Informativeness score
    """
    # Train a regressor to predict the entire factors from the entire latents
    model = GradientBoostingRegressor()
    model.fit(latents, factors)

    # Compute feature importances
    importance_matrix = np.abs(model.feature_importances_).reshape(-1, factors.shape[1])

    # Compute disentanglement
    disentanglement_scores = 1.0 - scipy.stats.entropy(importance_matrix.T + 1e-10, base=latents.shape[1])
    disentanglement = np.mean(disentanglement_scores)

    # Compute completeness
    completeness_scores = 1.0 - scipy.stats.entropy(importance_matrix + 1e-10, base=factors.shape[1])
    completeness = np.mean(completeness_scores)

    # Compute informativeness (R^2 score)
    informativeness = model.score(latents, factors)

    return disentanglement, completeness, informativeness
    
def compute_sap(self, latents, factors, continuous_factors=True):
    """
    Compute SAP (Separated Attribute Predictability) - measures the difference in prediction error for the top two latent dimensions most predictive of each factor.
    latents: (N, D) latent representations (e.g., identity latents)
    factors: (N, K) ground truth factors (e.g., age latents)
    continuous_factors: Whether the factors are continuous (True) or discrete (False)
    """
    sap_scores = []
    for j in range(factors.shape[1]):  # Iterate over factors
        errors = []
        for i in range(latents.shape[1]):  # Iterate over latent dimensions
            model = LinearRegression() if continuous_factors else GradientBoostingRegressor()
            model.fit(latents[:, i].reshape(-1, 1), factors[:, j])
            predictions = model.predict(latents[:, i].reshape(-1, 1))
            error = mean_squared_error(factors[:, j], predictions)
            errors.append(error)
        errors = np.sort(errors)
        sap_scores.append(errors[1] - errors[0])  # Difference between top two errors
    return np.mean(sap_scores)

def compute_beta_vae_score(self, latents, factors):
    """
    Compute Beta-VAE Score - measures disentanglement by training a classifier to predict factors from latents.
    latents: (N, D) latent representations (e.g., identity latents)
    factors: (N, K) ground truth factors (e.g., age latents)
    """
    accuracies = []
    for j in range(factors.shape[1]):  # Iterate over factors
        model = LinearRegression()
        model.fit(latents, factors[:, j])
        predictions = model.predict(latents)
        accuracy = mean_squared_error(factors[:, j], predictions)
        accuracies.append(accuracy)
    return np.mean(accuracies)

def stats_tests_correlation_new(self, train_loader, val_loader, test_loader):

    # Your numpy arrays
    age_latents = np.random.rand(332, 9)  # Replace with your actual data
    identity_latents = np.random.rand(332, 45)  # Replace with your actual data

    # SAP Metric
    sap_results = sap_score.compute_sap(
        mus_test=identity_latents,  # Learned representations
        ys=age_latents,             # Ground truth factors
        continuous_factors=True,     # Set to True if ground truth factors are continuous
        num_train=300,
        num_test=32
    )
    print("SAP Metric Results:", sap_results)

    # DCI Metric
    dci_results = dci.compute_dci(
        mus_test=identity_latents,  # Learned representations
        ys=age_latents              # Ground truth factors
    )
    print("DCI Metric Results:", dci_results)

def stats_tests_correlation(self, train_loader, val_loader, test_loader):
    """
    Perform statistical tests to check if age is disentangled from feature latents.

    """

    # process_data outputs: feature_latents, age_latents, gt_ages, gt_ages_norm, data_dataset
    train_identity_latents, train_age_latents, train_gt_age, train_gt_age_norm, _ = self.process_data(train_loader, datasets=None, diagonal=True)
    val_identity_latents, val_age_latents, val_gt_age, val_gt_age_norm, _ = self.process_data(val_loader, datasets=None, diagonal=True)
    test_identity_latents, test_age_latents, test_gt_age, test_gt_age_norm, _ = self.process_data(test_loader, datasets=None, diagonal=True)

    ####### DISENTANGLEMENT_LIB PYTORCH #######

    # from disentanglement_lib.mig import _compute_mig
    # from disentanglement_lib.dci import _compute_dci
    from disentanglement_lib.sap_score import _compute_sap

    identity_latents_train_val = np.concatenate((train_identity_latents, val_identity_latents), axis=0)
    identity_latents_test = test_identity_latents
    age_latents_train_val = np.concatenate((train_age_latents, val_age_latents), axis=0)
    age_latents_test = test_age_latents
    gt_ages_train_val = np.concatenate((train_gt_age, val_gt_age), axis=0)
    gt_ages_test = test_gt_age
    gt_ages_norm_train_val = np.concatenate((train_gt_age_norm, val_gt_age_norm), axis=0)
    gt_ages_norm_test = test_gt_age_norm

    ####### SAP and DCI SCORES #######
    # ------------------------------------------------------------------
    # 1) Cross-latent global SAP (age in id / id in age)
    # ------------------------------------------------------------------

    sap_score = _compute_sap(identity_latents_train_val.T, age_latents_train_val.T, identity_latents_test.T, age_latents_test.T, continuous_factors=True)
    print("SAP Score (age in id):", sap_score)
    sap_score = list(sap_score.values())[0]
    # self.log['test/sap_score_age_in_id'] = sap_score
    sap_score = _compute_sap(age_latents_train_val.T, identity_latents_train_val.T, age_latents_test.T, identity_latents_test.T, continuous_factors=True)
    print("SAP Score (id in age):", sap_score)
    sap_score = list(sap_score.values())[0]
    # self.log['test/sap_score_id_in_age'] = sap_score

    # dci_score = _compute_dci(identity_latents_train_val.T, age_latents_train_val.T, identity_latents_test.T, age_latents_test.T)
    # dci_score = dci_score["disentanglement"]
    # print("DCI Score (age in id):", dci_score)
    # # self.log['test/dci_d_score_age_in_id'] = dci_score
    # dci_score = _compute_dci(age_latents_train_val.T, identity_latents_train_val.T, age_latents_test.T, identity_latents_test.T)
    # dci_score = dci_score["disentanglement"]
    # print("DCI Score (id in age):", dci_score)
    # # self.log['test/dci_d_score_id_in_age'] = dci_score

    # ------------------------------------------------------------------
    # 2) Proper SAP vs *ground-truth age* (using diagonal only)
    # ------------------------------------------------------------------

    # Build train+val and test sets for identity latents vs gt age
    # id_train_val = identity_latents_train_val
    # id_test = identity_latents_test

    # gt_age_train_val = np.concatenate((train_gt_ages, val_gt_ages), axis=0)  # shape (N_tv, ?) or (N_tv,)
    # gt_age_test = test_gt_ages  # shape (N_test, ?)

    # Ensure ages are (N, 1) then transpose -> (1, N) for _compute_sap
    gt_ages_norm_train_val = gt_ages_norm_train_val.reshape(-1, 1)
    gt_ages_norm_test = gt_ages_norm_test.reshape(-1, 1)

    # mus: [num_latents, num_samples], ys: [num_factors=1, num_samples]
    sap_age_in_id_gt = _compute_sap(
        identity_latents_train_val.T,          # mus
        gt_ages_norm_train_val.T,      # ys (GT age as factor)
        identity_latents_test.T,               # mus_test
        gt_ages_norm_test.T,           # ys_test
        continuous_factors=True,
    )
    print("SAP Score (GT age in id latents):", sap_age_in_id_gt)
    # sap_age_in_id_gt_val = list(sap_age_in_id_gt.values())[0]
    # self.log["test/sap_score_gt_age_in_id"] = sap_age_in_id_gt_val

    # Optionally: SAP of GT age vs *age latents* as well (should be high if age block is good)
    # age_train_val_all = age_latents_train_val
    # age_test_all = age_latents_test

    sap_age_in_age_gt = _compute_sap(
        age_latents_train_val.T,
        gt_ages_norm_train_val.T,
        age_latents_test.T,
        gt_ages_norm_test.T,
        continuous_factors=True,
    )
    print("SAP Score (GT age in age latents):", sap_age_in_age_gt)
    # sap_age_in_age_gt_val = list(sap_age_in_age_gt.values())[0]
    # self.log["test/sap_score_gt_age_in_age"] = sap_age_in_age_gt_val


    # === FEATURE-LEVEL R_2 TESTS ===
    # ------------------------------------------------------------------
    # 3) Feature-level R² (age in id / id in age) - train linear regression models for each feature block and compute R² on test set
    # ------------------------------------------------------------------

    # feature_sap_results_age_in_id = {}
    # feature_dci_results_id_in_age = {}
    feature_r2_results_id_in_age = {}
    feature_r2_results_age_in_id = {}
    features = ["Temporal", "Eyes", "Cheekbones", "Cheeks", "Jaw", "Forehead", "Chin", "Lips", "Nose"]

    # Assuming 45 id latents (5 per feature) and 9 age latents
    num_features = age_latents_train_val.shape[1]
    id_latent_size = identity_latents_train_val.shape[1]
    id_per_feature = id_latent_size // num_features

    model_age_in_id = LinearRegression()
    model_id_in_age = LinearRegression()

    for i in range(num_features):
        id_inds = list(range(i * id_per_feature, (i + 1) * id_per_feature))
        age_ind = i

        feature_name = features[i]
        id_train_sub = identity_latents_train_val[:, id_inds]
        id_test_sub = identity_latents_test[:, id_inds]
        age_train_sub = age_latents_train_val[:, age_ind]
        age_test_sub = age_latents_test[:, age_ind]

        age_train_sub = age_train_sub.reshape(-1, 1)
        age_test_sub = age_test_sub.reshape(-1, 1)

        # sap_feat_age_in_id = _compute_sap(id_train_sub.T, age_train_sub.T,
        #                         id_test_sub.T, age_test_sub.T,
        #                         continuous_factors=True)
        # sap_feat_val_age_in_id = list(sap_feat_age_in_id.values())[0]
        # feature_sap_results_age_in_id[feature_name] = sap_feat_val_age_in_id
        # self.log[f"test/feature_sap_age_in_id/{feature_name}"] = sap_feat_val_age_in_id
        # print(f"{feature_name} (age in id): SAP={sap_feat_val_age_in_id}")

        # dci_feat_id_in_age = _compute_dci(age_train_sub.T, id_train_sub.T,
        #                         age_test_sub.T, id_test_sub.T)
        # # dci_feat_val_id_in_age = list(dci_feat_id_in_age.values())[0]
        # dci_feat_val_id_in_age = dci_feat_id_in_age["disentanglement"]
        # feature_dci_results_id_in_age[feature_name] = dci_feat_val_id_in_age
        # # self.log[f"test/feature_sap_id_in_age/{feature_name}"] = sap_feat_val_id_in_age
        # # print(f"{feature_name} (id in age): DCI Disentanglement={dci_feat_val_id_in_age}")

        # Fit the model for "age in identity"
        model_age_in_id.fit(id_train_sub, age_train_sub)
        r2 = r2_score(age_test_sub, model_age_in_id.predict(id_test_sub), multioutput='variance_weighted')
        feature_r2_results_age_in_id[feature_name] = r2

        # Fit the model for "identity in age"
        model_id_in_age.fit(age_train_sub, id_train_sub)
        r2 = r2_score(id_test_sub, model_id_in_age.predict(age_test_sub), multioutput='variance_weighted')
        feature_r2_results_id_in_age[feature_name] = r2

    # print("Feature-level SAP results (age in id):", feature_sap_results_age_in_id)
    # print("Feature-level DCI results (id in age):", feature_dci_results_id_in_age)
    print("Feature-level R² results (age in id):", feature_r2_results_age_in_id)
    print("Feature-level R² results (id in age):", feature_r2_results_id_in_age)
    self.log["test/feature_r2_age_in_id"] = feature_r2_results_age_in_id
    self.log["test/feature_r2_id_in_age"] = feature_r2_results_id_in_age

    ####### MI SCORE #######

    # # Concatenate the feature latents and age labels from train, val, and test sets
    # identity_latents = np.concatenate((train_identity_latents, val_identity_latents, test_identity_latents), axis=0)
    # age_latents = np.concatenate((train_age_latents, val_age_latents, test_age_latents), axis=0)
    # # gt_age_norms = np.concatenate((train_gt_age_norm, val_gt_age_norm, test_gt_age_norm), axis=0)

    # # mig_score = _compute_mig(identity_latents.T, age_latents.T)
    # # print("MIG Score (id vs age):", mig_score)
    # # mig_score = _compute_mig(age_latents.T, identity_latents.T)
    # # print("MIG Score (age vs id):", mig_score)

    # # # not from dis_lib
    # # # MI: How much age information is in identity latents
    # # mi_matrix = np.zeros((identity_latents.shape[1], age_latents.shape[1]))
    # # for i in range(identity_latents.shape[1]):
    # #     for j in range(age_latents.shape[1]):
    # #         mi_matrix[i, j] = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents[:, j]).mean()
    # # mi_score= np.mean(mi_matrix)
    # # print("MI Score:", mi_score)
    # # mi_score = float(mi_score)
    # # self.log["test/mi_score"].log([mi_score]) 


    ####### DISENTANGLEMENT_LIB TENSORFLOW #######

    # # Import gin and disentanglement_lib utilities
    # import gin
    # import disentanglement_lib.evaluation.metrics.utils as disentanglement_utils

    # # Register make_discretizer as an external configurable
    # gin.external_configurable(disentanglement_utils.make_discretizer, module='disentanglement_lib.utils')

    # # Set gin configuration for MIG
    # gin.bind_parameter('discretizer.num_bins', 10)
    # gin.bind_parameter('discretizer.discretizer_fn', disentanglement_utils.make_discretizer)

    # # === STEP 1: Debugging Latent Shapes ===
    # print("Identity latents shape:", identity_latents.shape)
    # print("Age latents shape:", age_latents.shape)

    # # === STEP 2: Dataset Wrapper ===
    # class SimpleDataset:
    #     def __init__(self, factors):
    #         self.factors = factors
    #         self.num_samples = self.factors.shape[0]

    #     def sample(self, num_samples, random_state):
    #         # Ensure we don't sample more than the available data
    #         if num_samples > self.num_samples:
    #             num_samples = self.num_samples
    #         idx = random_state.choice(self.num_samples, num_samples, replace=False)
    #         obs = idx[:, None]  # obs contains sample indices
    #         print("Sampled indices:", idx)  # Debugging
    #         print("Sampled factors shape:", self.factors[idx].shape)  # Debugging
    #         return obs, self.factors[idx]

    # # Create datasets for age and identity factors
    # identity_factors_dataset = SimpleDataset(identity_latents)
    # age_factors_dataset = SimpleDataset(age_latents)

    # # === STEP 3: Representation Function ===
    # def make_repr_fn(latents, name=""):
    #     def fn(obs):
    #         indices = obs[:, 0].astype(int)
    #         rep = latents[indices]
    #         print(f"[{name}] Representation shape: {rep.shape}")  # Debugging
    #         return rep
    #     return fn

    # # Create representation functions
    # identity_repr_fn = make_repr_fn(identity_latents, name="identity")
    # age_repr_fn = make_repr_fn(age_latents, name="age")

    # # === STEP 4: Compute Metrics ===
    # random_state = np.random.RandomState(0)
    # num_samples = identity_latents.shape[0]
    # num_train = int(num_samples * 0.8)
    # num_test = num_samples - num_train

    # results = {}

    # print("Debugging MIG Inputs:")
    # print("Age factors dataset shape:", age_factors_dataset.factors.shape)
    # print("Number of training samples:", num_train)
    # print("Random state:", random_state)

    # # --- Age info in Identity latents ---
    # results["mig_identity_age"] = mig.compute_mig(
    #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train
    # )['discrete_mig']
    # results["dci_identity_age"] = dci.compute_dci(
    #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train, num_test=num_test
    # )['disentanglement']
    # results["sap_identity_age"] = sap_score.compute_sap(
    #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train, num_test=num_test
    # )['SAP_score']
    # results["beta_identity_age"] = beta_vae.compute_beta_vae_sklearn(
    #     age_factors_dataset, identity_repr_fn, random_state, num_train=num_train, num_eval=num_test
    # )['eval_accuracy']

    # # --- Identity info in Age latents ---
    # results["mig_age_identity"] = mig.compute_mig(
    #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train
    # )['discrete_mig']
    # results["dci_age_identity"] = dci.compute_dci(
    #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train, num_test=num_test
    # )['disentanglement']
    # results["sap_age_identity"] = sap_score.compute_sap(
    #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train, num_test=num_test
    # )['SAP_score']
    # results["beta_age_identity"] = beta_vae.compute_beta_vae_sklearn(
    #     identity_factors_dataset, age_repr_fn, random_state, num_train=num_train, num_eval=num_test
    # )['eval_accuracy']

    # # === STEP 5: Print Results ===
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")

    # # Log results
    # for k, v in results.items():
    #     self.log[f'test/{k}'] = v

    # ####### NEW IMPLEMENTATION #######

    # # age_min, age_max = map(int, self._config['data']['dataset_age_range'].split('-'))

    # # # Test 1: Age Information in Identity Latents
    # # print("Running Test 1: Age Information in Identity Latents")
    # # mig_score = self.compute_mi(identity_latents, age_latents)
    # # print("MIG Score (Age in Identity):", mig_score)

    # # dci_scores_dis, dci_scores_comp, dci_scores_info = self.compute_dci(identity_latents, age_latents)
    # # print("DCI Scores Disentanglement (Age in Identity):", dci_scores_dis)
    # # print("DCI Scores Completeness (Age in Identity):", dci_scores_comp)
    # # print("DCI Scores Informativeness (Age in Identity):", dci_scores_info)

    # sap_score = self._compute_sap(identity_latents, age_latents, continuous_factors=True)
    # print("SAP Score (Age in Identity):", sap_score)

    # beta_vae_score = self.compute_beta_vae_score(identity_latents, age_latents)
    # print("Beta-VAE Score (Age in Identity):", beta_vae_score)

    # # self.log['test/mig_score_age_in_identity'] = str(mig_score)
    # # self.log['test/dci_score_age_in_identity'] = str(dci_scores)
    # # self.log['test/sap_score_age_in_identity'] = str(sap_score)
    # # self.log['test/beta_vae_score_age_in_identity'] = str(beta_vae_score)

    # # Test 2: Identity Information in Age Latents
    # print("Running Test 2: Identity Information in Age Latents")
    # mig_score = self.compute_mi(age_latents, identity_latents)
    # # dci_scores_dis, dci_scores_comp, dci_scores_info = self.compute_dci(age_latents, identity_latents)
    # sap_score = self.compute_sap(age_latents, identity_latents, continuous_factors=True)
    # beta_vae_score = self.compute_beta_vae_score(age_latents, identity_latents)

    # print("MIG Score (Identity in Age):", mig_score)
    # # print("DCI Scores Disentanglement (Identity in Age):", dci_scores_dis)
    # # print("DCI Scores Completeness (Age in Identity):", dci_scores_comp)
    # # print("DCI Scores Informativeness (Age in Identity):", dci_scores_info)
    # print("SAP Score (Identity in Age):", sap_score)
    # print("Beta-VAE Score (Identity in Age):", beta_vae_score)

    # # self.log['test/mig_score_identity_in_age'] = str(mig_score)
    # # self.log['test/dci_score_identity_in_age'] = str(dci_scores)
    # # self.log['test/sap_score_identity_in_age'] = str(sap_score)
    # # self.log['test/beta_vae_score_identity_in_age'] = str(beta_vae_score)

    ####### OLD IMPLEMENTATION #######

    # if self._config['data']['swap_features']:
    #     latents_per_feature = (self._manager.model_latent_size - self._config['model']['age_latent_size']) // len(self._manager.latent_regions)
    #     num_groups = identity_latents.shape[1] // latents_per_feature  # should give 9 groups

    # age_latents_size = self._config['model']['age_latent_size']

    # if age_latents_size == 1:

    #     sap_score = sap(factors=age_latents, codes=identity_latents, continuous_factors=True, nb_bins=10, regression=True)

    #     # Mutual Information (all feature latents vs single age latent)
    #     mi_all_all = mutual_info_regression(identity_latents, age_latents.ravel())

    #     if self._config['data']['swap_features']:
    
    #         # Mutual Information (each subset of feature latents vs single age latent)
    #         mi_subsets_single_age = []
    #         for i in range(num_groups):
    #             mi = mutual_info_regression(identity_latents[:, i*latents_per_feature:(i+1)*latents_per_feature], age_latents.ravel())
    #             mi_subsets_single_age.append(mi)
            
    #         # Mutual Information (each feature latent vs single age latent)
    #         mi_each_feature_single_age = []
    #         for i in range(identity_latents.shape[1]):
    #             mi = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents.ravel())
    #             mi_each_feature_single_age.append(mi)
        
    # else:

    #     sap_score = sap(factors=age_latents, codes=identity_latents, continuous_factors=True, nb_bins=10, regression=True)

    #     # Mutual Information (all feature latents vs all age latents) 
    #     # compute the MI for each pair of shape latent and age latent.
    #     mi_matrix = np.zeros((identity_latents.shape[1], age_latents.shape[1]))

    #     for i in range(identity_latents.shape[1]): 
    #         for j in range(age_latents.shape[1]):  
    #             mi_matrix[i, j] = mutual_info_regression(identity_latents[:, i].reshape(-1, 1), age_latents[:, j]).mean()

    #     mi_all_all = np.mean(mi_matrix)

    #     if self._config['data']['swap_features']:

    #         # Mutual Information (each subset of feature latents vs each single age latent)
    #         mi_subsets_each_age = []
    #         for i in range(num_groups):
    #             mi = mutual_info_regression(identity_latents[:, i*latents_per_feature:(i+1)*latents_per_feature], age_latents[:, i]).mean()
    #             mi_subsets_each_age.append(mi)
    #         mi_subsets_each_age_mean = np.array(mi_subsets_each_age).mean()
    #         mi_subsets_each_age = [[mi_subsets_each_age_mean], mi_subsets_each_age]
            
    #         # Mutual Information (each feature latent vs corresponding age latent)
    #         mi_each_feature_each_age = []
    #         for i in range(num_groups):
    #             for j in range(latents_per_feature):
    #                 mi = mutual_info_regression(identity_latents[:, i*latents_per_feature + j].reshape(-1, 1), age_latents[:, i])
    #                 mi_each_feature_each_age.append(mi[0])
    #         mi_each_feature_each_age_mean = np.array(mi_each_feature_each_age).mean()
    #         mi_each_feature_each_age = [[mi_each_feature_each_age_mean], mi_each_feature_each_age]

    # # All values desired to be low
    # print("- SAP Score:")
    # print(sap_score)

    # print("- Mutual Information between all shape_latents and all age_latents:")
    # print(mi_all_all)

    # self.log['test/sap'] = str(sap_score)
    # self.log['test/mi_all_all'] = str(mi_all_all)

    # if self._config['data']['swap_features'] and age_latents_size != 1:
    #     self.log['test/mi_subsets_each_age'] = str(mi_subsets_each_age)
    #     self.log['test/mi_each_feature_each_age'] = str(mi_each_feature_each_age)


