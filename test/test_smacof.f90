program test_smacof
 use sklearn_common,only:dp,require
 use sklearn_smacof,only:smacof,smacof_result
 use sklearn_mds,only:mds
 use,intrinsic::ieee_exceptions
 implicit none
 type(smacof_result)::fit,again
 type(mds)::model
 real(dp)::d(3,3),initial(3,1),draw1(4),draw2(4)
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 integer,allocatable::state(:)
 real(dp),allocatable::mapped(:,:)
 integer::size_
 d=reshape([0._dp,1._dp,3._dp,1._dp,0._dp,2._dp,3._dp,2._dp,0._dp],[3,3])
 initial(:,1)=[-1._dp,0._dp,1._dp]
 fit=smacof(d,init=initial,max_iter=1)
 call require(size(fit%embedding,2)==1.and.fit%n_init_effective==1,'SMACOF initial dimensions and restart override')
 call require(maxval(abs(fit%embedding(:,1)-[-4._dp/3,-1._dp/3,5._dp/3]))<1.e-12_dp,'SMACOF Guttman step')
 call require(abs(fit%stress-2)<1.e-12_dp.and.fit%n_iter==1,'SMACOF pre-update stress')
 call random_seed(size=size_);allocate(state(size_));call random_seed(get=state);call random_number(draw1)
 call random_seed(put=state)
 call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
 fit=smacof(d,random_state=42,n_init=3)
 call ieee_get_halting_mode(ieee_all,after_modes);call ieee_get_flag(ieee_all,after_flags)
 call require(all(modes.eqv.after_modes).and.all(flags.eqv.after_flags),'SMACOF preserves IEEE state')
 call random_number(draw2);call require(all(draw1==draw2),'SMACOF preserves global RNG')
 again=smacof(d,random_state=42,n_init=3)
 call require(all(fit%embedding==again%embedding).and.fit%stress==again%stress,'SMACOF private seeded repeatability')
 fit=smacof(d,metric=.false.,normalized_stress=.true.,max_iter=3,random_state=7)
 call require(fit%normalized_stress.and.fit%n_iter<=3,'SMACOF logical normalization overload')
 call random_seed(get=state);call random_number(draw1);call random_seed(put=state)
 model%dissimilarity='precomputed';model%n_init=1;model%max_iter=1;model%random_state=42
 mapped=model%fit_transform(d)
 call random_number(draw2);call require(all(draw1==draw2),'MDS estimator preserves global RNG')
 fit=smacof(d,n_init=1,max_iter=1,random_state=42)
 call require(all(mapped==fit%embedding).and.model%stress_==fit%stress,'MDS and functional seeded starts agree')
 d=0;fit=smacof(d,n_init=1,max_iter=4)
 call require(fit%stress==0.and.fit%n_iter==4.and.all(fit%embedding==0),'SMACOF Source zero-stress iteration behavior')
 print *,'SMACOF metric/nonmetric fitting, owned results and private RNG passed'
end program
