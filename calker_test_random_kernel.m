function calker_test_random_kernel(proj_name, exp_name, ker, events)


    % loading labels
    calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', ker.proj_dir, proj_name, exp_name, ker.feat, ker.suffix);

	calker_common_exp_dir = sprintf('%s/%s/experiments/%s-calker/common/%s', ker.proj_dir, proj_name, exp_name, ker.feat);

	test_db_file = sprintf('database_%s.mat', ker.test_pat);

	db_file = fullfile(calker_common_exp_dir, test_db_file);

	fprintf('Loading database [%s]...\n', db_file);
    load(db_file, 'database');


    n_event = length(events);

    all_labels = zeros(n_event, length(database.label));

    for ii = 1:length(database.label),
        for jj = 1:n_event,
            if database.label(ii) == jj,
                all_labels(jj, ii) = 1;
            else
                all_labels(jj, ii) = -1;
            end
        end
    end


    % number test kf
    %n_test_kf = size(all_labels, 2);
	
	n_test_kf = size(database.video, 2);	%% Update Sep 6, 2013
    fprintf('Number test kf %d\n', n_test_kf);

    num_part = ceil(n_test_kf/ker.chunk_size);
    cols = fix(linspace(1, n_test_kf + 1, num_part+1));
		
	for rr = 1:ker.numrand,
		
		scorePath = sprintf('%s/r-scores/%d/%s/%s.r%d.scores.mat', calker_exp_dir, ker.randim, ker.test_pat, ker.name, rr);
		
		if checkFile(scorePath), 
			fprintf('Skipped testing %s \n', scorePath);
			continue;
		end
		
		models = struct;
		scores = struct;
		
		for jj = 1:n_event,
			event_name = events{jj};
			
			%modelPath = sprintf('%s/models/%s.%s.%s.model.mat', calker_exp_dir, event_name, ker.name, ker.type);
			modelPath = sprintf('%s/r-models/%d/%s.%s.%s.model.%d.mat', calker_exp_dir, ker.randim, event_name, ker.name, ker.type, rr);
			
			if ~checkFile(modelPath),
				error('Model not found %s \n', modelPath);			
			end
			
			fprintf('Loading model ''%s''...\n', event_name);
			models.(event_name) = load(modelPath);
			tmp_scores{jj} = cell(num_part, 1);
			scores.(event_name) = [];
		end
		
			%load test partition
		for kk = 1:num_part,
			sel = [cols(kk):cols(kk+1)-1];
			part_name = sprintf('%s_%d_%d', ker.testname, cols(kk), cols(kk+1)-1);
			%kerPath = sprintf('%s/kernels/%s/%s.%s.mat', calker_exp_dir, ker.test_pat, part_name, ker.type);
			kerPath = sprintf('%s/r-kernels/%s/%d/%s.%s.r%d.mat', calker_exp_dir, ker.test_pat, ker.randim, part_name, ker.type, rr);
			
			fprintf('Loading kernel %s ...\n', kerPath); 
			kernels_ = load(kerPath) ;
			base = kernels_.matrix;
			%info = whos('base') ;
			%fprintf('\tKernel matrices size %.2f GB\n', info.bytes / 1024^3) ;
			
			% Nt = # test
			% N  = # train
			[N, Nt] = size(base) ;

			parfor jj = 1:n_event,
				event_name = events{jj};
				fprintf('-- [%d/%d] -- Testing event ''%s''...\n', kk, num_part, event_name);
				%only test at svind
				%test_base = base(models.(event_name).svind,:);
				%sub_scores = models.(event_name).alphay' * test_base + models.(event_name).b;
				
				[y, acc, dec] = svmpredict(zeros(Nt, 1), [(1:Nt)' base'], models.(event_name).libsvm_cl, '-b 1') ;		
				sub_scores = dec(:, 1)';
				
				tmp_scores{jj}{kk} = sub_scores;
			end
			
			clear base;
		end
		
		for jj = 1:n_event,
			event_name = events{jj};
			scores.(event_name) = cat(2, tmp_scores{jj}{:});
		end
			
		%saving scores
		fprintf('\tSaving scores ''%s''.\n', scorePath) ;
		
		output_score_dir = fileparts(scorePath);
		if ~exist(output_score_dir, 'file'),
			mkdir(output_score_dir);
		end
		
		ssave(scorePath, '-STRUCT', 'scores') ;
		
		%fprintf('\tCalculating maps ''%s''.\n', scorePath) ;
		%calker_cal_map(proj_name, exp_name, ker, events);
	end

end