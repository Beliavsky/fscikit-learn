program reference_skewed_features_driver
  use sklearn_common,only: dp
  use sklearn_kernel_approximation,only: skewed_chi2_sampler
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(skewed_chi2_sampler)::model
  real(dp),allocatable::x(:,:),q(:,:),z(:,:),query(:,:),original_x(:,:),original_q(:,:),first(:,:),offset(:)
  integer,allocatable::before(:),after(:)
  integer::c,cases,n,p,m,k,seed,mode,mutate,ns
  real(dp)::skew,query_skew
  call random_seed(size=ns);allocate(before(ns),after(ns))
  read(*,*) cases
  do c=1,cases
    read(*,*) n,p,m,k,seed,skew,query_skew,mutate,mode
    allocate(x(n,p),q(m,p));read(*,*) x;read(*,*) q
    model=skewed_chi2_sampler();model%n_components=k;model%random_state=seed;model%skewedness=skew
    if(mode==1) query=model%transform(q)
    if(mode==2) x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
    if(mode==3) x(1,1)=ieee_value(0._dp,ieee_positive_inf)
    if(mode==4) q(1,1)=ieee_value(0._dp,ieee_quiet_nan)
    if(mode==5) q(1,1)=ieee_value(0._dp,ieee_positive_inf)
    original_x=x;original_q=q;call random_seed(get=before)
    if(mode==7) then
      call model%fit(x)
    else
      z=model%fit_transform(x)
    end if
    if(mode==6) then
      query=model%transform(q(:,:p-1))
    else
      if(mutate>0) model%n_components=mutate
      model%skewedness=query_skew;query=model%transform(q)
    end if
    call random_seed(get=after)
    print *, 'BEGIN',c
    call emit('weights',reshape(model%random_weights_,[size(model%random_weights_)]))
    call emit('offset',model%random_offset_)
    if(mode==7) then
      call emit('train',[real(dp)::])
    else
      call emit('train',reshape(z,[size(z)]))
    end if
    call emit('query',reshape(query,[size(query)]))
    call emit('metadata',real([model%n_features_in_,size(model%random_weights_,2),model%n_components],dp))
    call emit('preserved',real([merge(1,0,all(transfer(x,[0_int64])==transfer(original_x,[0_int64]))), &
      merge(1,0,all(transfer(q,[0_int64])==transfer(original_q,[0_int64]))),merge(1,0,all(before==after))],dp))
    first=model%random_weights_;offset=model%random_offset_;model%n_components=k;model%skewedness=skew
    call model%fit(x)
    call emit('repeat',real([merge(1,0,all(transfer(first,[0_int64])==transfer(model%random_weights_,[0_int64]))), &
      merge(1,0,all(transfer(offset,[0_int64])==transfer(model%random_offset_,[0_int64])))],dp))
    print *, 'END',c
    deallocate(x,q)
  end do
contains
 subroutine emit(name,values)
  character(*),intent(in)::name
  real(dp),intent(in)::values(:)
  print *,name,size(values)
  write(*,'(*(es25.16e3,1x))') values
 end subroutine
end program
