
function calker_cal_train_random_kernel(proj_name, exp_name, ker)

	feature_ext = ker.feat;

	calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', ker.proj_dir, proj_name, exp_name, ker.feat, ker.suffix);

	kerPath = sprintf('%s/kernels/%s/%s', calker_exp_dir, ker.dev_pat, ker.devname);

	devHistPath = sprintf('%s/kernels/%s/%s.mat', calker_exp_dir, ker.dev_pat, ker.histName);
	
	scaleParamsPath = sprintf('%s/kernels/%s/%s.mat', calker_exp_dir, ker.dev_pat, ker.scaleparamsName);
	
	log2g_list = ker.startG:ker.stepG:ker.endG;
	numLog2g = length(log2g_list);

	fprintf('\tLoading devel features for kernel %s ... \n', feature_ext) ;
	if exist(devHistPath),
		load(devHistPath);
	else
		dev_hists = calker_load_traindata(proj_name, exp_name, ker);
		
		if ker.feature_scale == 1,	
			fprintf('Feature scaling...\n');	
			[dev_hists, scale_params] = calker_feature_scale(dev_hists);	
			save(scaleParamsPath, 'scale_params');		
		end
		
		fprintf('\tSaving devel features for kernel %s ... \n', feature_ext) ;
		save(devHistPath, 'dev_hists', '-v7.3');
		
	end
	
	
	for rr = 1:ker.numrand,
		
		ridx = randperm(size(dev_hists, 1));
		
		ridx = ridx(1:ker.randim);
		randidx_Path = sprintf('%s/r-kernels/%s/%d/%s.randindex.r%d.mat', calker_exp_dir, ker.dev_pat, ker.randim, ker.devname, rr);
		idx_save(randidx_Path, randidx_Path);
		
		if ker.cross,
			parfor jj = 1:numLog2g,
				cv_ker = ker;
				log2g = log2g_list(jj);
				gamma = 2^log2g;	
				cv_ker.mu = gamma;
				%cv_kerPath = sprintf('%s.gamma%s.mat', kerPath, num2str(gamma));
				cv_kerPath = sprintf('%s/r-kernels/%s/%d/%s.gamma%s.r%d.mat', calker_exp_dir, ker.dev_pat, ker.randim, ker.devname, num2str(gamma), rr);
				
				if ~exist(cv_kerPath),
					fprintf('\tCalculating devel kernel %s with gamma = %f... \n', feature_ext, gamma) ;	
					cv_ker = calcKernel(cv_ker, dev_hists(ridx, :));
					
					fprintf('\tSaving kernel ''%s''.\n', cv_kerPath) ;
					par_save( cv_kerPath, cv_ker );
				else
					fprintf('Skipped calculating kernel [%s]...\n', cv_kerPath);
				end			
			end
		else
			%heu_kerPath = sprintf('%s.heuristic.mat', kerPath);
			heu_kerPath = sprintf('%s/r-kernels/%s/%d/%s.heuristic.r%d.mat', calker_exp_dir, ker.dev_pat, ker.randim, ker.devname, rr);
			if ~exist(heu_kerPath),
				fprintf('\t[ randim = %d, randnum = %d] Calculating devel kernel %s with heuristic gamma ... \n', ker.randim, rr, feature_ext) ;	
				ker = calcKernel(ker, dev_hists(ridx, :));
				
				fprintf('\tSaving kernel ''%s''.\n', heu_kerPath) ;
				par_save( heu_kerPath, ker );
			else
				fprintf('Skipped calculating kernel [%s]...\n', heu_kerPath);
			end		
		end
	end
	
end

function par_save( output_file, ker )
	output_dir = fileparts(output_file);
	if ~exist(output_dir, 'file'), mkdir(output_dir); end;
	ssave(output_file, '-STRUCT', 'ker', '-v7.3');
end

function idx_save( output_file, index )
	output_dir = fileparts(output_file);
	if ~exist(output_dir, 'file'), mkdir(output_dir); end;
	save( output_file, 'index');
end
