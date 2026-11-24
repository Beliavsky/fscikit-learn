program reference_power_edges_driver
 use sklearn,only:dp,power_transformer,power_transform
 implicit none
 type(power_transformer)::model
 real(dp),allocatable::x(:,:),q(:,:),z(:,:)
 real(dp)::lambda
 integer::n,p,m,op,std,i
 character(16)::method
 read(*,*) n,p,m,op,std,method,lambda
 allocate(x(n,p),q(m,p))
 do i=1,n
  read(*,*) x(i,:)
 end do
 do i=1,m
  read(*,*) q(i,:)
 end do
 model%method=method;model%standardize=std==1
 select case(op)
 case(0)
  z=model%fit_transform(x)
  call emit('lambda',model%lambdas_);call emit('train',reshape(z,[size(z)]))
  z=model%transform(q);call emit('query',reshape(z,[size(z)]))
 case(1)
  z=power_transform(x,method,std==1);call emit('free',reshape(z,[size(z)]))
 case(2,3)
  call model%fit(x);model%lambdas_=lambda
  if(op==2) then
   z=model%transform(q)
  else
   z=model%inverse_transform(q)
  end if
  call emit('scalar',reshape(z,[size(z)]))
 end select
contains
 subroutine emit(name,v)
  character(*),intent(in)::name
  real(dp),intent(in)::v(:)
  write(*,'(a,*(1x,es25.16e3))') name,v
 end subroutine
end program
