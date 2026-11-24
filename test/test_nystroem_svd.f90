program test_nystroem_svd
 use sklearn_common,only:dp,require
 use sklearn_kernel_approximation,only:nystroem
 use sklearn_nystroem_svd,only:kernel_svd
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite
 implicit none
 type(nystroem)::model
 real(dp)::x(5,3),a(3,3),u(3,3),v(3,3),s(3),identity(3,3)
 real(dp),allocatable::z(:,:),first(:,:)
 integer,allocatable::before(:),after(:)
 integer::i,j,status,n
 do i=1,5
  x(i,:)=[real(i,dp),sin(real(i,dp)),cos(real(i,dp))]
 end do
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 model%n_components=3;z=model%fit_transform(x)
 call require(all(model%component_indices_==[3,1,2]),'Nystroem NumPy landmark permutation')
 first=z;model%gamma=.7_dp;z=model%transform(x)
 call require(any(abs(first-z)>1.e-8_dp),'Nystroem transform uses live kernel controls')
 model%kernel='sigmoid';model%coef0=-1;z=model%fit_transform(x)
 call require(all(ieee_is_finite(model%normalization_)),'Nystroem indefinite basis SVD')
 model=nystroem();model%n_components=3;model%kernel='callable';model%kernel_callback=>callback
 z=model%fit_transform(x);call require(all(shape(z)==[5,3]),'Nystroem typed callback')
 a=0;a(1,1)=1.e10_dp;a(2,2)=1.e-8_dp;a(3,3)=-2
 call kernel_svd(a,u,s,v,status,240)
 call require(status==0.and.s(3)>0.and.abs(s(3)-1.e-8_dp)<1.e-22_dp,'SVD preserves small nonzero singular values')
 identity=0
 do i=1,3
  identity(i,i)=1
 end do
 call require(maxval(abs(matmul(transpose(u),u)-identity))<1.e-12_dp,'SVD complete left directions')
 call require(maxval(abs(matmul(u*spread(s,1,3),transpose(v))-a))<1.e-12_dp,'SVD reconstruction')
 model=nystroem();model%n_components=3;model%kernel='precomputed'
 z=model%fit_transform(a);call require(all(ieee_is_finite(z)),'Nystroem precomputed full-rank basis')
 a=0;call kernel_svd(a,u,s,v,status,240)
 call require(status==0.and.all(u==identity).and.all(v==identity).and.all(s==0),'zero SVD identity completion')
 call random_seed(get=after);call require(all(before==after),'Nystroem fits preserve intrinsic RNG')
 print *, 'Nystroem seeds, live kernels, indefinite/precomputed bases and complete SVD: PASS'
contains
 function callback(a,b) result(value)
  real(dp),intent(in)::a(:),b(:)
  real(dp)::value
  value=exp(-.3_dp*sum((a-b)**2))
 end function
end program
