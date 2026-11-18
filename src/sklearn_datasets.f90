module sklearn_datasets
  use sklearn_common
  use sklearn_model_selection, only: shuffle_indices
  implicit none
  private
  public :: make_blobs,make_regression,make_moons,make_circles
contains
  subroutine make_blobs(n_samples,n_features,n_centers,x,y,random_state,cluster_std,centers)
    integer, intent(in) :: n_samples,n_features,n_centers
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    integer, intent(in), optional :: random_state
    real(dp), intent(in), optional :: cluster_std,centers(:,:)
    real(dp) :: std
    real(dp), allocatable :: means(:,:)
    integer, allocatable :: order(:)
    integer :: i,c
    call require(n_samples>0.and.n_features>0.and.n_centers>0,'invalid blob sizes')
    std=1; if(present(cluster_std)) std=cluster_std
    call require(std>=0,'cluster_std must be nonnegative')
    if(present(random_state)) call seed_rng(random_state)
    allocate(means(n_centers,n_features))
    if(present(centers)) then
      call require(all(shape(centers)==[n_centers,n_features]),'blob centers shape mismatch'); means=centers
    else
      call random_number(means); means=20*means-10
    end if
    x=normal_random(n_samples,n_features)*std; allocate(y(n_samples))
    do i=1,n_samples
      c=1+mod(i-1,n_centers); y(i)=c-1; x(i,:)=x(i,:)+means(c,:)
    end do
    order=shuffle_indices(n_samples); x=x(order,:); y=y(order)
  end subroutine
  subroutine make_regression(n_samples,n_features,x,y,n_informative,noise,bias,random_state,coef)
    integer, intent(in) :: n_samples,n_features
    real(dp), allocatable, intent(out) :: x(:,:),y(:)
    integer, intent(in), optional :: n_informative,random_state
    real(dp), intent(in), optional :: noise,bias
    real(dp), allocatable, intent(out), optional :: coef(:)
    real(dp) :: coefficients(n_features),std,offset
    real(dp), allocatable :: errors(:,:)
    integer :: informative
    call require(n_samples>0.and.n_features>0,'invalid regression sizes')
    informative=min(10,n_features); if(present(n_informative)) informative=n_informative
    call require(informative>=0.and.informative<=n_features,'invalid informative feature count')
    std=0; offset=0
    if(present(noise)) std=noise
    if(present(bias)) offset=bias
    call require(std>=0,'noise must be nonnegative')
    if(present(random_state)) call seed_rng(random_state)
    x=normal_random(n_samples,n_features); coefficients=0
    call random_number(coefficients(:informative)); coefficients=100*coefficients
    errors=normal_random(n_samples,1); y=matmul(x,coefficients)+offset+std*errors(:,1)
    if(present(coef)) coef=coefficients
  end subroutine
  subroutine make_moons(n_samples,x,y,noise,random_state,shuffle)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), intent(in), optional :: noise
    integer, intent(in), optional :: random_state
    logical, intent(in), optional :: shuffle
    integer :: i,outer,inner
    real(dp) :: theta,std
    logical :: shuf
    integer, allocatable :: order(:)
    call require(n_samples>=4,'make_moons needs at least four samples')
    std=0; if(present(noise)) std=noise
    shuf=.true.; if(present(shuffle)) shuf=shuffle
    call require(std>=0,'noise must be nonnegative')
    if(present(random_state)) call seed_rng(random_state)
    outer=n_samples/2; inner=n_samples-outer; allocate(x(n_samples,2),y(n_samples))
    do i=1,outer
      theta=acos(-1._dp)*(i-1)/(outer-1); x(i,:)=[cos(theta),sin(theta)]; y(i)=0
    end do
    do i=1,inner
      theta=acos(-1._dp)*(i-1)/(inner-1)
      x(outer+i,:)=[1-cos(theta),0.5_dp-sin(theta)]; y(outer+i)=1
    end do
    if(shuf) then
      order=shuffle_indices(n_samples); x=x(order,:); y=y(order)
    end if
    if(std>0) x=x+std*normal_random(n_samples,2)
  end subroutine
  subroutine make_circles(n_samples,x,y,noise,factor,random_state,shuffle)
    integer, intent(in) :: n_samples
    real(dp), allocatable, intent(out) :: x(:,:)
    integer, allocatable, intent(out) :: y(:)
    real(dp), intent(in), optional :: noise,factor
    integer, intent(in), optional :: random_state
    logical, intent(in), optional :: shuffle
    integer :: i,outer,inner
    real(dp) :: theta,std,f
    logical :: shuf
    integer, allocatable :: order(:)
    call require(n_samples>=4,'make_circles needs at least four samples')
    std=0; f=0.8_dp
    if(present(noise)) std=noise
    if(present(factor)) f=factor
    shuf=.true.; if(present(shuffle)) shuf=shuffle
    call require(std>=0.and.f>0.and.f<1,'invalid circles noise or factor')
    if(present(random_state)) call seed_rng(random_state)
    outer=n_samples/2; inner=n_samples-outer; allocate(x(n_samples,2),y(n_samples))
    do i=1,outer
      theta=2*acos(-1._dp)*(i-1)/outer; x(i,:)=[cos(theta),sin(theta)]; y(i)=0
    end do
    do i=1,inner
      theta=2*acos(-1._dp)*(i-1)/inner
      x(outer+i,:)=f*[cos(theta),sin(theta)]; y(outer+i)=1
    end do
    if(shuf) then
      order=shuffle_indices(n_samples); x=x(order,:); y=y(order)
    end if
    if(std>0) x=x+std*normal_random(n_samples,2)
  end subroutine
end module
