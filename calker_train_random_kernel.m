function calker_train_random_kernel(proj_name, exp_name, ker, events)

    test_on_train = 0;
	
	calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', ker.proj_dir, proj_name, exp_name, ker.feat, ker.suffix);

    traindb_file = fullfile(calker_exp_dir, 'metadata', 'traindb.mat');
	
    load(traindb_file, 'traindb');

    % event names
 
    n_event = length(events);

    all_labels = zeros(n_event, length(traindb.selected_label));

    for ii = 1:length(traindb.selected_label),
        for jj = 1:n_event,
            if traindb.selected_label(ii) == jj,
                all_labels(jj, ii) = 1;
            else
                all_labels(jj, ii) = -1;
            end
        end
    end

    kerPath = sprintf('%s/kernels/%s/%s', calker_exp_dir, ker.dev_pat, ker.devname);
	
	parfor kk = 1:n_event,
		event_name = events{kk};
			
		fprintf('Training event ''%s''...\n', event_name);	
		
		labels = double(all_labels(kk,:));
		posWeight = ceil(length(find(labels == -1))/length(find(labels == 1)));
		
		log2g_list = ker.startG:ker.stepG:ker.endG;
		numLog2g = length(log2g_list);
		
		for rr = ker.numrand,
		
			if ker.cross,
				svm = cell(numLog2g, 1);
				maxacc = cell(numLog2g, 1);
				
				for jj = 1:numLog2g,
					cv_ker = ker;
					log2g = log2g_list(jj);
					gamma = 2^log2g;	
					
					cv_kerPath = sprintf('%s.gamma%s.mat', kerPath, num2str(gamma));
					fprintf('Loading kernel %s ...\n', cv_kerPath); 
					kernels_ = load(cv_kerPath) ;
					base = kernels_.matrix;

					fprintf('SVM learning with predefined kernel matrix...\n');
					[svm_, maxacc_] = calker_svmkernellearn(base, labels,   ...
									   'type', 'C',        ...
									   ...%'C', 10,            ...
									   'verbosity', 0,     ...
									   ...%'rbf', 1,           ...
									   'crossvalidation', 5, ...
									   'weights', [+1 posWeight ; -1 1]') ;
					fprintf(' cur acc = %f, at gamma = %f...\n', maxacc_, gamma);
					
					svm{jj} = svm_;
					maxacc{jj} = maxacc_;
					
				end
				
				maxacc = cat(1, maxacc{:});
				[~, max_idx] = 	max(maxacc);
				svm = svm{max_idx};
				gamma = 2^log2g_list(max_idx);
				fprintf(' best acc = %f, at gamma = %f...\n', maxacc(max_idx), gamma);
				
			else
						
				modelPath = sprintf('%s/r-models/%d/%s.%s.%s.model.%d.mat', calker_exp_dir, ker.randim, event_name, ker.name, ker.type, rr);
		
				if checkFile(modelPath),
					fprintf('Skipped training %s \n', modelPath);
					continue;
				end
		
				heu_kerPath = sprintf('%s/r-kernels/%s/%d/%s.heuristic.r%d.mat', calker_exp_dir, ker.dev_pat, ker.randim, ker.devname, rr);
				%heu_kerPath = sprintf('%s.heuristic.mat', kerPath);
				
				fprintf('Loading kernel %s ...\n', heu_kerPath); 
				kernels_ = load(heu_kerPath) ;
				base = kernels_.matrix;
				
				fprintf('SVM learning with predefined kernel matrix...\n');
				svm = calker_svmkernellearn(base, labels,   ...
								   'type', 'C',        ...
								   ...%'C', 10,            ...
								   'verbosity', 0,     ...
								   ...%'rbf', 1,           ...
								   'crossvalidation', 5, ...
								   'weights', [+1 posWeight ; -1 1]') ;
								   
				if isfield(kernels_, 'mu'),
					gamma = kernels_.mu;
				end
				
				svm = svmflip(svm, labels);

				if strcmp(ker.type, 'echi2'),
					svm.gamma = gamma;
				end
				
				
				%clear kernels_;
			end
				
			fprintf('\tSaving model ''%s''.\n', modelPath) ;
			par_save( modelPath, svm );			
		end

	end
	
end

function par_save( modelPath, svm )
	ssave(modelPath, '-STRUCT', 'svm') ;
end
