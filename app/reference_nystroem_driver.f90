program reference_nystroem_driver
 use sklearn_common,only:dp
 use sklearn_kernel_approximation,only:nystroem
 use sklearn_nystroem_svd,only:kernel_svd
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
 implicit none
 type(nystroem)::model
 real(dp),allocatable::x(:,:),q(:,:),g(:,:),z(:,:),query(:,:),u(:,:),v(:,:),s(:),first(:,:),original(:,:),original_q(:,:)
 integer,allocatable::before(:),after(:)
 integer::cases,c,n,p,m,k,seed,kind,degree,status,ns,mutate,mode
 real(dp)::gamma,coef0,query_gamma
 character(16),parameter::kernels(*)=[character(16)::'rbf','linear','poly','polynomial','cosine', &
   'laplacian','sigmoid','chi2','additive_chi2','precomputed','callable','callable']
 call random_seed(size=ns);allocate(before(ns),after(ns))
 read(*,*) cases
 do c=1,cases
  read(*,*) n,p,m,k,seed,kind,gamma,coef0,degree,query_gamma,mutate,mode
  allocate(x(n,p),q(m,p),g(min(n,k),min(n,k)));read(*,*) x;read(*,*) q;read(*,*) g
  model=nystroem();model%n_components=k;model%random_state=seed;model%kernel='unknown'
  if(kind>=0.and.kind<size(kernels)) model%kernel=kernels(kind+1)
  model%gamma=gamma;model%coef0=coef0;model%degree=degree
  if(kind==10) model%kernel_callback=>asymmetric_callback
  if(kind==11) model%kernel_callback=>rbf_callback
  if(mode==1) query=model%transform(q)
  if(mode==2) x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==3) x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==4) q(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  if(mode==5) q(1,1)=ieee_value(0._dp,ieee_positive_inf)
  if(mode==7) model%kernel_callback=>bad_callback
  if(mode==8) nullify(model%kernel_callback)
  original=x;original_q=q;call random_seed(get=before);z=model%fit_transform(x)
  model%gamma=query_gamma;if(mutate>0) model%n_components=mutate
  if(mode==6) then
    query=model%transform(q(:,:p-1))
  else
    query=model%transform(q)
  end if
  call random_seed(get=after)
  print *,'BEGIN',c
  call emit('indices',real(model%component_indices_,dp))
  call emit('components',reshape(model%components_,[size(model%components_)]))
  call emit('normalization',reshape(model%normalization_,[size(model%normalization_)]))
  call emit('norm_gram',reshape(matmul(transpose(model%normalization_),model%normalization_), &
    [size(model%normalization_)]))
  call emit('train',reshape(z,[size(z)]));call emit('query',reshape(query,[size(query)]))
  call emit('train_gram',reshape(matmul(z,transpose(z)),[size(z,1)**2]))
  call emit('query_gram',reshape(matmul(query,transpose(query)),[size(query,1)**2]))
  call emit('cross_gram',reshape(matmul(z,transpose(query)),[size(z,1)*size(query,1)]))
  call emit('metadata',real([model%n_features_in_,size(model%components_,1),model%n_components],dp))
  call emit('preserved',real([merge(1,0,all(original==x)),merge(1,0,all(original_q==q)),merge(1,0,all(before==after))],dp))
  first=model%normalization_;model%gamma=gamma;model%n_components=k;call model%fit(x)
  call emit('repeat',real([merge(1,0,all(first==model%normalization_))],dp))
  allocate(u(size(g,1),size(g,1)),v(size(g,1),size(g,1)),s(size(g,1)))
  call kernel_svd(g,u,s,v,status,240)
  call emit('svd_status',[real(status,dp)]);call emit('singular',s)
  call emit('svd_u',reshape(u,[size(u)]));call emit('svd_v',reshape(v,[size(v)]))
  print *,'END',c
  deallocate(x,q,g,u,v,s)
 end do
contains
 function bad_callback(a,b) result(value)
  real(dp),intent(in)::a(:),b(:)
  real(dp)::value
  value=ieee_value(0._dp,ieee_quiet_nan)
 end function
 function asymmetric_callback(a,b) result(value)
  real(dp),intent(in)::a(:),b(:)
  real(dp)::value
  value=dot_product(a,b)+.2_dp*a(1)-.1_dp*b(1)+1
 end function
 function rbf_callback(a,b) result(value)
  real(dp),intent(in)::a(:),b(:)
  real(dp)::value
  value=exp(-.3_dp*sum((a-b)**2))
 end function
 subroutine emit(name,values)
  character(*),intent(in)::name
  real(dp),intent(in)::values(:)
  print *,name,size(values)
  write(*,'(*(es25.16e3,1x))') values
 end subroutine
end program
