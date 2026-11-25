program reference_covariance_functions_driver
  use sklearn_common,only:dp
  use sklearn_covariance,only:empirical_covariance,shrunk_covariance,ledoit_wolf,oas,covariance_estimate
  implicit none
  real(dp),allocatable::x(:,:),cov(:,:),batch(:,:,:),out(:,:,:)
  type(covariance_estimate)::estimate
  integer::mode,n,p,center,block,rank,ios,i,j
  real(dp)::amount
  real(dp),allocatable::a4(:,:,:,:),b4(:,:,:,:)
  real(dp),allocatable::a5(:,:,:,:,:),b5(:,:,:,:,:)
  real(dp),allocatable::a6(:,:,:,:,:,:),b6(:,:,:,:,:,:)
  real(dp),allocatable::a7(:,:,:,:,:,:,:),b7(:,:,:,:,:,:,:)
  real(dp),allocatable::a8(:,:,:,:,:,:,:,:),b8(:,:,:,:,:,:,:,:)
  real(dp),allocatable::a9(:,:,:,:,:,:,:,:,:),b9(:,:,:,:,:,:,:,:,:)
  real(dp),allocatable::a10(:,:,:,:,:,:,:,:,:,:),b10(:,:,:,:,:,:,:,:,:,:)
  real(dp),allocatable::a11(:,:,:,:,:,:,:,:,:,:,:),b11(:,:,:,:,:,:,:,:,:,:,:)
  real(dp),allocatable::a12(:,:,:,:,:,:,:,:,:,:,:,:),b12(:,:,:,:,:,:,:,:,:,:,:,:)
  real(dp),allocatable::a13(:,:,:,:,:,:,:,:,:,:,:,:,:),b13(:,:,:,:,:,:,:,:,:,:,:,:,:)
  real(dp),allocatable::a14(:,:,:,:,:,:,:,:,:,:,:,:,:,:),b14(:,:,:,:,:,:,:,:,:,:,:,:,:,:)
  real(dp),allocatable::a15(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:),b15(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:)
  do
    read(*,*,iostat=ios)mode,n,p,center,block,rank,amount
    if(ios/=0)exit
    if(mode<3)then
      allocate(x(n,p))
      do i=1,n
        read(*,*)x(i,:)
      end do
      select case(mode)
      case(0)
        if(rank==1)then
          cov=empirical_covariance(x(1,:),assume_centered=center==1)
        else
          cov=empirical_covariance(x,assume_centered=center==1)
        end if
        amount=-1
      case(1)
        estimate=ledoit_wolf(x,assume_centered=center==1,block_size=block)
        cov=estimate%covariance;amount=estimate%shrinkage
      case(2)
        estimate=oas(x,assume_centered=center==1)
        cov=estimate%covariance;amount=estimate%shrinkage
      end select
      write(*,*)size(cov,1),size(cov,2)
      write(*,'(*(es26.17e3,1x))')cov
      write(*,'(es26.17e3)')amount
      deallocate(x)
    else
      allocate(batch(n,p,p))
      do i=1,n
        do j=1,p
          read(*,*)batch(i,j,:)
        end do
      end do
      select case(rank)
      case(2)
        cov=shrunk_covariance(batch(1,:,:),amount);out=reshape(cov,[1,p,p])
      case(3)
        out=shrunk_covariance(batch,amount)
      case(4)
        a4=reshape(batch,[n,1,p,p]);b4=shrunk_covariance(a4,amount)
        out=reshape(b4,[n,p,p]);deallocate(a4,b4)
      case(5)
        a5=reshape(batch,[n,1,1,p,p]);b5=shrunk_covariance(a5,amount)
        out=reshape(b5,[n,p,p]);deallocate(a5,b5)
      case(6)
        a6=reshape(batch,[n,1,1,1,p,p]);b6=shrunk_covariance(a6,amount)
        out=reshape(b6,[n,p,p]);deallocate(a6,b6)
      case(7)
        a7=reshape(batch,[n,1,1,1,1,p,p]);b7=shrunk_covariance(a7,amount)
        out=reshape(b7,[n,p,p]);deallocate(a7,b7)
      case(8)
        a8=reshape(batch,[n,1,1,1,1,1,p,p]);b8=shrunk_covariance(a8,amount)
        out=reshape(b8,[n,p,p]);deallocate(a8,b8)
      case(9)
        a9=reshape(batch,[n,1,1,1,1,1,1,p,p]);b9=shrunk_covariance(a9,amount)
        out=reshape(b9,[n,p,p]);deallocate(a9,b9)
      case(10)
        a10=reshape(batch,[n,1,1,1,1,1,1,1,p,p]);b10=shrunk_covariance(a10,amount)
        out=reshape(b10,[n,p,p]);deallocate(a10,b10)
      case(11)
        a11=reshape(batch,[n,1,1,1,1,1,1,1,1,p,p]);b11=shrunk_covariance(a11,amount)
        out=reshape(b11,[n,p,p]);deallocate(a11,b11)
      case(12)
        a12=reshape(batch,[n,1,1,1,1,1,1,1,1,1,p,p]);b12=shrunk_covariance(a12,amount)
        out=reshape(b12,[n,p,p]);deallocate(a12,b12)
      case(13)
        a13=reshape(batch,[n,1,1,1,1,1,1,1,1,1,1,p,p]);b13=shrunk_covariance(a13,amount)
        out=reshape(b13,[n,p,p]);deallocate(a13,b13)
      case(14)
        a14=reshape(batch,[n,1,1,1,1,1,1,1,1,1,1,1,p,p]);b14=shrunk_covariance(a14,amount)
        out=reshape(b14,[n,p,p]);deallocate(a14,b14)
      case(15)
        a15=reshape(batch,[n,1,1,1,1,1,1,1,1,1,1,1,1,p,p]);b15=shrunk_covariance(a15,amount)
        out=reshape(b15,[n,p,p]);deallocate(a15,b15)
      end select
      write(*,*)size(out,1),size(out,2),size(out,3)
      write(*,'(*(es26.17e3,1x))')out
      write(*,'(es26.17e3)')amount
      deallocate(batch)
    end if
  end do
end program
