program test_fastica_function
 use sklearn_common,only:dp,require
 use sklearn_fast_ica,only:fastica,fastica_result,fast_ica
 use,intrinsic::ieee_exceptions
 implicit none
 type(fastica_result)::fit,again
 type(fast_ica)::model
 real(dp)::x(80,2),draw1(3),draw2(3)
 integer::i,n
 integer,allocatable::state(:)
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 do i=1,80
  x(i,:)=[sin(real(i,dp))+.2_dp*cos(3._dp*i),cos(3._dp*i)+2]
 end do
 call random_seed(size=n);allocate(state(n));call random_seed(get=state)
 call random_number(draw1);call random_seed(put=state)
 call ieee_set_flag(ieee_all,.false.)
 call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
 fit=fastica(x,random_state=42,whiten_solver='eigh',tol=1.e-8_dp)
 call ieee_get_halting_mode(ieee_all,after_modes);call ieee_get_flag(ieee_all,after_flags)
 call require(all(modes.eqv.after_modes).and.all(flags.eqv.after_flags),'FastICA preserves IEEE state')
 call random_number(draw2);call require(all(draw1==draw2),'FastICA preserves global RNG')
 again=fastica(x,random_state=42,whiten_solver='eigh',tol=1.e-8_dp)
 call require(all(fit%sources==again%sources),'private seeded FastICA repeats exactly')
 call require(all(shape(fit%whitening)==[2,2]).and.all(shape(fit%unmixing)==[2,2]),'owned ICA matrices')
 call require(maxval(abs(sum(fit%sources**2,dim=1)/80-1))<1.e-10_dp,'functional unit source variance')
 again=fastica(x,whiten='none',compute_sources=.false.,max_iter=1)
 call require(.not.allocated(again%sources).and..not.allocated(again%mean).and. &
  .not.allocated(again%whitening),'Source None results use unallocated fields')
 model%whiten_solver='eigh';model%random_state=42;model%tol=1.e-8_dp;call model%fit(x)
 call require(all(model%unmixing_==fit%unmixing),'functional and fitted ICA share the unmixing matrix')
 fit=fastica(x,fun='cube',max_iter=1,random_state=7)
 again=fastica(x,contrast_proc=cube_contrast,max_iter=1,random_state=7)
 call require(maxval(abs(fit%unmixing-again%unmixing))<1.e-10_dp,'typed callable contrast matches cube')
 print *,'Functional ICA, eigh whitening, private RNG and owned results passed'
contains
 subroutine cube_contrast(z,g,gprime)
  real(dp),intent(in)::z(:,:)
  real(dp),intent(out)::g(:,:),gprime(:)
  integer::row
  g=z**3
  do row=1,size(z,1)
   gprime(row)=sum(3*z(row,:)**2)/size(z,2)
  end do
 end subroutine
end program
