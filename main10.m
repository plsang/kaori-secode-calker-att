
function main10(proj_name, exp_name, kf_name, feature_ext, feat_dim, ker_type, skip_convert, cross)

addpath('/net/per900a/raid0/plsang/tools/kaori-secode-calker/support');
addpath('/net/per900a/raid0/plsang/tools/libsvm-3.12/matlab');
addpath('/net/per900a/raid0/plsang/tools/vlfeat-0.9.13/toolbox');
addpath('/net/per900a/raid0/plsang/tools/kaori-secode-med10');

% run vl_setup with no prefix
vl_setup('noprefix');

%proj_name = 'trecvidmed10';
%exp_name = 'trecvidmed10-100000';
%kf_name = 'keyframe-100000';

events = {'assembling_shelter', 'batting_in_run', 'making_cake'};


%feature_ext = 'densetrajectory.mbh.Soft-4000-VL2.MBH.trecvidmed10.devel.fc.l2';
%feature_ext = 'densetrajectory.mbh.Soft-4000-VL2.MBH.trecvidmed10.devel.kcb.l2';
%feature_ext = 'dense6.sift.Soft-500-VL2.trecvidmed10.devel.kcb.l2';
%feature_ext = 'mfcc.Soft-4000-VL2.trecvidmed10.devel.kcb.l2';
%feature_ext = 'densetrajectory.full.Soft-4000-VL2.trecvidmed10.devel.kcb.l2';
%feature_ext = 'dense6.sift.Soft-500-VL2.trecvidmed10.devel.spm.kcb.l2';
%feature_ext = 'dense6.sift.Soft-4000-VL2.trecvidmed10.devel.spm.kcb.l2';

calker_exp_dir = sprintf('/net/per900a/raid0/plsang/%s/experiments/%s-calker/%s', proj_name, exp_name, feature_ext);
if ~exist(calker_exp_dir, 'file'),
	mkdir(calker_exp_dir);
	mkdir(fullfile(calker_exp_dir, 'metadata'));
	mkdir(fullfile(calker_exp_dir, 'kernels'));
	mkdir(fullfile(calker_exp_dir, 'scores'));
	mkdir(fullfile(calker_exp_dir, 'models'));
end

if nargin < 5,
	feat_dim = 4000;
	ker_type = 'echi2';
	skip_convert = 1;
end

ker = calker_build_kerdb(proj_name, exp_name, feature_ext, ker_type, feat_dim, cross);

if ~skip_convert,
	if matlabpool('size') < 1, matlabpool open; end;
	convert_feature( kf_name, feature_ext(1:end-3), 'devel', 'l2', feat_dim );
	convert_feature( kf_name, feature_ext(1:end-3), 'test', 'l2', feat_dim );
	matlabpool close
end

if matlabpool('size') < 1, matlabpool open; end;
calker_create_database(proj_name, exp_name, kf_name, ker);
calker_create_traindb(proj_name, exp_name, ker);
calker_cal_train_kernel(proj_name, exp_name, ker);
calker_train_kernel(proj_name, exp_name, ker, events);
calker_cal_test_kernel(proj_name, exp_name, ker, events);
calker_test_kernel(proj_name, exp_name, ker, events);
calker_cal_map(proj_name, exp_name, ker, events);
matlabpool close