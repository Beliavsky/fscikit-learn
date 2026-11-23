program reference_geometric_datasets_driver
  use sklearn_common, only: dp
  use sklearn_datasets, only: make_blobs,make_moons,make_circles
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  integer :: cases,c,kind,form,n,p,k,seed,shuf,noise_kind,center_mode,std_mode,box_mode,i
  integer, allocatable :: counts(:),labels(:),again(:),nc
  real(dp) :: noise,factor,std,lo,hi,nan
  real(dp), allocatable :: centers(:,:),box(:),scales(:),x(:,:),other(:,:),returned(:,:)
  character(64) :: invalid
  nan=ieee_value(0._dp,ieee_quiet_nan)
  call get_command_argument(1,invalid)
  if(len_trim(invalid)>0) then
    call reject(trim(invalid));error stop 'invalid geometric generator input accepted'
  end if
  read(*,*) cases
  do c=1,cases
    read(*,*) kind,form,n,p,k,seed,shuf,noise_kind,center_mode,std_mode,box_mode
    read(*,*) noise,factor,std,lo,hi
    allocate(counts(k),scales(k));read(*,*) counts;read(*,*) scales
    if(center_mode==1.or.center_mode==2) then
      allocate(centers(k,p))
      do i=1,k
        read(*,*) centers(i,:)
      end do
    end if
    if(box_mode/=0) then
      allocate(box(2));box=[lo,hi]
      if(box_mode==2) box=[nan,ieee_value(0._dp,ieee_positive_inf)]
    end if
    if(center_mode/=3) allocate(nc,source=k)
    write(*,'(a,1x,i0)') 'BEGIN',c
    if(kind==0) then
      select case(std_mode)
      case(0);call blobs_call()
      case(1);call blobs_call(std)
      case(2);call blobs_call(scales)
      end select
      call matrix('centers',returned)
      call vector('metadata',real([size(x,1),size(x,2),size(returned,1)],dp))
    else
      call curves_call()
      call vector('metadata',real([size(x,1),size(x,2),2],dp))
    end if
    call matrix('x',x);call vector('labels',real(labels,dp))
    call matrix('repeated_x',other);call vector('repeated_labels',real(again,dp))
    write(*,'(a)') 'END';flush(6)
    deallocate(counts,scales)
    if(allocated(centers)) deallocate(centers)
    if(allocated(box)) deallocate(box)
    if(allocated(nc)) deallocate(nc)
  end do
contains
  subroutine blobs_call(scale)
    real(dp), optional, intent(in) :: scale(..)
    integer :: features
    features=p;if(center_mode==2) features=p+5
    if(form==0) then
      call make_blobs(n,features,nc,x,labels,seed,scale,centers,box,shuf==1,returned)
      call make_blobs(n,features,nc,other,again,seed,scale,centers,box,shuf==1)
    else
      call make_blobs(counts,features,nc,x,labels,seed,scale,centers,box,shuf==1,returned)
      call make_blobs(counts,features,nc,other,again,seed,scale,centers,box,shuf==1)
    end if
  end subroutine
  subroutine curves_call()
    real(dp), allocatable :: supplied_noise
    if(noise_kind/=0) allocate(supplied_noise,source=noise)
    if(kind==1) then
      if(form==0) then
        call make_moons(n,x,labels,supplied_noise,seed,shuf==1)
        call make_moons(n,other,again,supplied_noise,seed,shuf==1)
      else
        call make_moons(counts,x,labels,supplied_noise,seed,shuf==1)
        call make_moons(counts,other,again,supplied_noise,seed,shuf==1)
      end if
    else
      if(form==0) then
        call make_circles(n,x,labels,supplied_noise,factor,seed,shuf==1)
        call make_circles(n,other,again,supplied_noise,factor,seed,shuf==1)
      else
        call make_circles(counts,x,labels,supplied_noise,factor,seed,shuf==1)
        call make_circles(counts,other,again,supplied_noise,factor,seed,shuf==1)
      end if
    end if
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    integer :: j
    write(*,'(a,1x,i0)',advance='no') name,size(a)
    do j=1,size(a)
      write(*,'(1x,es25.17e3)',advance='no') a(j)
    end do
    write(*,*)
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(transpose(a),[size(a)]))
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: bad_std(2,2),bad_centers(2,2)
    bad_std=1;bad_centers=0
    select case(name)
    case('blob_samples');call make_blobs(0,2,2,x,labels)
    case('blob_features');call make_blobs(2,0,2,x,labels)
    case('blob_centers');call make_blobs(2,2,0,x,labels)
    case('empty_counts');call make_blobs([integer::],2,x=x,y=labels)
    case('negative_counts');call make_blobs([2,-1],2,x=x,y=labels)
    case('count_overflow');call make_blobs([huge(1),1],2,x=x,y=labels)
    case('count_centers');call make_blobs([2,3],2,3,x,labels)
    case('center_rows');call make_blobs(2,2,3,x,labels,centers=bad_centers)
    case('center_features');call make_blobs(2,2,2,x,labels,centers=bad_centers(:,:0))
    case('nan_centers');bad_centers(1,1)=nan;call make_blobs(2,2,2,x,labels,centers=bad_centers)
    case('negative_std');call make_blobs(2,2,2,x,labels,cluster_std=-1._dp)
    case('nan_std');call make_blobs(2,2,2,x,labels,cluster_std=nan)
    case('inf_std');call make_blobs(2,2,2,x,labels,cluster_std=ieee_value(0._dp,ieee_positive_inf))
    case('vector_std');call make_blobs(2,2,2,x,labels,cluster_std=[1._dp])
    case('negative_vector_std');call make_blobs(2,2,2,x,labels,cluster_std=[0._dp,-1._dp])
    case('matrix_std');call make_blobs(2,2,2,x,labels,cluster_std=bad_std)
    case('box_size');call make_blobs(2,2,2,x,labels,center_box=[1._dp])
    case('box_order');call make_blobs(2,2,2,x,labels,center_box=[2._dp,1._dp])
    case('box_nan');call make_blobs(2,2,2,x,labels,center_box=[nan,1._dp])
    case('box_overflow');call make_blobs(2,2,2,x,labels,center_box=[-1.e308_dp,1.e308_dp])
    case('blob_seed');call make_blobs(2,2,2,x,labels,random_state=-1)
    case('moon_samples');call make_moons(0,x,labels)
    case('circle_samples');call make_circles(0,x,labels)
    case('moon_counts');call make_moons([1,2,3],x,labels)
    case('circle_counts');call make_circles([1],x,labels)
    case('curve_negative');call make_moons([1,-1],x,labels)
    case('curve_noise');call make_moons(2,x,labels,noise=-1._dp)
    case('curve_nan');call make_circles(2,x,labels,noise=nan)
    case('curve_inf');call make_moons(2,x,labels,noise=ieee_value(0._dp,ieee_positive_inf))
    case('factor_one');call make_circles(2,x,labels,factor=1._dp)
    case('factor_negative');call make_circles(2,x,labels,factor=-.1_dp)
    case('factor_nan');call make_circles(2,x,labels,factor=nan)
    case('curve_seed');call make_moons(2,x,labels,random_state=-1)
    case default;error stop 'unknown rejection'
    end select
  end subroutine
end program
