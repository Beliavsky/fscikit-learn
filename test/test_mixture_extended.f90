program test_mixture_extended
 use sklearn_common,only:dp,require
 use sklearn_mixture,only:gaussian_mixture
 use,intrinsic::ieee_exceptions
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite
 implicit none
 type(gaussian_mixture)::model,repeat,two
 real(dp)::x(30,2),identity(2,2),sample_mean(2),sample_cov(2,2),centered(2),scale(2,2)
 real(dp),allocatable::draws(:,:),again(:,:)
 integer,allocatable::labels(:),labels2(:),fits(:)
 integer::i,method
 integer,allocatable::rng_before(:),rng_after(:)
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 character(20),parameter::inits(4)=[character(20)::'kmeans','random','random_from_data','k-means++']
 do i=1,30
  x(i,:)=[sin(real(i,dp)),cos(2._dp*i)]+merge(0._dp,5._dp,i<=15)
 end do
 call random_seed(size=i);allocate(rng_before(i),rng_after(i));call random_seed(get=rng_before)
 do method=1,4
  model%n_components=2;model%init_params=inits(method);model%n_init=3;model%max_iter=3
  repeat=model
  call ieee_set_flag(ieee_all,.false.);call ieee_get_flag(ieee_all,flags)
  call ieee_get_halting_mode(ieee_all,modes);fits=model%fit_predict(x)
  call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'mixture fit_predict preserves IEEE state')
  call ieee_set_flag(ieee_all,.false.);call model%fit(x)
  call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
  call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'mixture fit preserves IEEE state')
  call repeat%fit(x)
  call require(maxval(abs(model%means_-repeat%means_))==0,'mixture initialization repeatability')
  call require(all(fits==model%predict(x)),'fit_predict final labels')
 end do
 model%init_params='random';model%max_iter=1;model%n_init=1;model%warm_start=.false.
 two=model;two%max_iter=2;call two%fit(x)
 call model%fit(x);model%warm_start=.true.;call model%fit(x)
 call require(maxval(abs(model%means_-two%means_))<1.e-12_dp,'warm mixture continues the EM trajectory')
 model%warm_start=.false.;model%max_iter=0;model%n_init=3;call model%fit(x)
 call require(model%n_iter_==0.and..not.model%converged_,'zero-iteration initialization metadata')
 call require(.not.ieee_is_finite(model%lower_bound_),'zero-iteration lower bound is negative infinity')
 model%max_iter=3;model%covariance_type='diag';call model%fit(x)
 call model%sample(53,draws,labels);call model%sample(53,again,labels2)
 call require(all(draws==again).and.all(labels==labels2),'mixture samples repeat for integer seed')
 call require(all(labels(2:)>=labels(:size(labels)-1)),'mixture samples are grouped by component')
 call random_seed(get=rng_after);call require(all(rng_before==rng_after),'mixture preserves global Fortran RNG')
 model%n_components=1;model%covariance_type='full';call model%fit(x)
 identity=matmul(model%covariances_(:,:,1),model%precisions_(:,:,1));identity(1,1)=identity(1,1)-1
 identity(2,2)=identity(2,2)-1
 call require(maxval(abs(identity))<1.e-10_dp,'mixture covariance and Cholesky precision')
 call ieee_set_flag(ieee_all,.false.);call model%sample(30000,draws,labels)
 call ieee_get_flag(ieee_all,after_flags);call ieee_get_halting_mode(ieee_all,after_modes)
 call require(all(flags.eqv.after_flags).and.all(modes.eqv.after_modes),'mixture sample preserves IEEE state')
 sample_mean=sum(draws,dim=1)/size(draws,1);sample_cov=0
 do i=1,size(draws,1)
  centered=draws(i,:)-sample_mean
  sample_cov=sample_cov+spread(centered,2,2)*spread(centered,1,2)
 end do
 sample_cov=sample_cov/size(draws,1)
 scale=sqrt(spread([model%covariances_(1,1,1),model%covariances_(2,2,1)],1,2)* &
   spread([model%covariances_(1,1,1),model%covariances_(2,2,1)],2,2))
 call require(all(abs(sample_cov-model%covariances_(:,:,1))<.08_dp*scale),'full-covariance sample moments')
 call require(maxval(abs(sample_mean-model%means_(1,:)))<.1_dp,'full-covariance sample mean')
 model%weights_init=[1._dp];allocate(model%means_init(1,2),model%precisions_init(2,2,1))
 model%means_init=0;model%precisions_init=0;model%precisions_init(1,1,1)=1;model%precisions_init(2,2,1)=1
 model%max_iter=1;call model%fit(x)
 call require(maxval(abs(model%means_(1,:)-sum(x,dim=1)/30))<1.e-12_dp,'explicit mixture starting parameters')
 call random_seed(get=rng_after);call require(all(rng_before==rng_after),'mixture sampling preserves global RNG')
 print *,'Mixture initialization, warm starts, fit_predict, diagnostics and sampling passed'
end program
